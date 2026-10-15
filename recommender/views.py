import logging
import math

from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    login,
    logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Count
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from .models import Prediction, UserProfile
from .ml import load_bundle, predict_one
from .weather import get_coordinates, fetch_climate_data


logger = logging.getLogger(__name__)

# Load the model once when Django imports this module.
# If unavailable, predictions fail safely.
BUNDLE = load_bundle()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def is_admin(user):
    """Return True only for authenticated staff users."""
    return user.is_authenticated and user.is_staff


def safe_redirect_url(request, url, fallback="predict"):
    """Prevent redirects to untrusted external websites."""
    if url and url_has_allowed_host_and_scheme(
        url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return redirect(url)

    return redirect(fallback)


def parse_number(post_data, field_name):
    """Read a required numeric field and reject invalid numbers."""
    raw_value = post_data.get(field_name, "").strip()

    if not raw_value:
        raise ValueError(f"Please enter a value for {field_name}.")

    try:
        value = float(raw_value)
    except (ValueError, TypeError):
        raise ValueError(
            f"{field_name} must be a valid number."
        ) from None

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number.")

    return value


def validate_crop_inputs(data):
    """Validate all measurements before predicting or saving."""

    required_fields = (
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    )

    # Ensure all required measurements are present and numeric.
    for field in required_fields:
        if field not in data:
            raise ValueError(
                "Please provide all required soil and climate values."
            )

        value = data[field]

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{field} must be a valid finite number."
            )

    # Reject an entirely zero-filled submission.
    if all(data[field] == 0 for field in required_fields):
        raise ValueError(
            "All inputs cannot be zero. "
            "Enter valid soil and climate values."
        )

    # Common Crop Recommendation dataset ranges.
    # Verify these against the dataset used to train your model.
    nutrient_limits = {
        "N": (0, 140),
        "P": (5, 145),
        "K": (5, 205),
    }

    for field, (minimum, maximum) in nutrient_limits.items():
        if not minimum <= data[field] <= maximum:
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
            )

    if not 8.8 <= data["temperature"] <= 43.7:
        raise ValueError(
            "Temperature must be between 8.8°C and 43.7°C."
        )

    if not 14.2 <= data["humidity"] <= 100:
        raise ValueError(
            "Humidity must be between 14.2% and 100%."
        )

    if not 3.5 <= data["ph"] <= 9.94:
        raise ValueError(
            "Soil pH must be between 3.5 and 9.94."
        )

    if not 20 <= data["rainfall"] <= 300:
        raise ValueError(
            "Rainfall must be between 20 and 300 mm."
        )

    return data


def save_prediction_if_logged_in(request, inputs, crop):
    """Save predictions only for authenticated, non-staff users."""

    if (
        not request.user.is_authenticated
        or request.user.is_staff
        or request.user.is_superuser
    ):
        return

    Prediction.objects.create(
        user=request.user,
        N=inputs["N"],
        P=inputs["P"],
        K=inputs["K"],
        temperature=inputs["temperature"],
        humidity=inputs["humidity"],
        ph=inputs["ph"],
        rainfall=inputs["rainfall"],
        predicted_label=crop,
    )


def get_prediction_bundle():
    """Use the loaded model or attempt to load it again."""

    if BUNDLE is not None:
        return BUNDLE

    return load_bundle()


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

def home(request):
    """Display the public project overview landing page."""
    return render(request, "recommender/home.html")


# ---------------------------------------------------------
# USER SIGNUP
# ---------------------------------------------------------

def signup_view(request):
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect("admin_dashboard")
        return redirect("predict")

    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        phone = request.POST.get("phone_number", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        try:
            if not full_name:
                raise ValueError("Please enter your full name.")

            if len(full_name) > 150:
                raise ValueError(
                    "Full name must be 150 characters or fewer."
                )

            if not email or "@" not in email:
                raise ValueError("Enter a valid email address.")

            if len(email) > 254:
                raise ValueError("Email address is too long.")

            if len(phone) > 20:
                raise ValueError(
                    "Phone number must be 20 characters or fewer."
                )

            if User.objects.filter(email__iexact=email).exists():
                raise ValueError(
                    "An account with that email already exists."
                )

            if User.objects.filter(username__iexact=email).exists():
                raise ValueError(
                    "An account with that email already exists."
                )

            if not password:
                raise ValueError("Password is required.")

            if len(password) < 8:
                raise ValueError(
                    "Password must contain at least 8 characters."
                )

            if confirm_password != password:
                raise ValueError("Passwords do not match.")

            if request.POST.get("agree") not in (
                "yes",
                "on",
                "true",
                "1",
            ):
                raise ValueError(
                    "Please agree to the terms before creating an account."
                )

            name_parts = full_name.split(maxsplit=1)

            # Create the user and profile atomically.
            with transaction.atomic():
                user = User.objects.create_user(
                    username=email,
                    email=email,
                    password=password,
                    first_name=name_parts[0],
                    last_name=(
                        name_parts[1] if len(name_parts) > 1 else ""
                    ),
                )

                UserProfile.objects.create(
                    user=user,
                    phone=phone,
                )

            messages.success(
                request,
                "Your account was created successfully. Please log in.",
            )
            return redirect("login")

        except ValueError as exc:
            messages.error(request, str(exc))

    return render(request, "recommender/signup.html")


# ---------------------------------------------------------
# USER LOGIN
# ---------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect("admin_dashboard")
        return redirect("predict")

    next_url = request.POST.get("next") or request.GET.get("next", "")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        user = authenticate(
            request,
            username=username,
            password=password,
        )

        # Support login using email for existing accounts.
        if user is None and username:
            matching_user = User.objects.filter(
                email__iexact=username
            ).first()

            if matching_user:
                user = authenticate(
                    request,
                    username=matching_user.username,
                    password=password,
                )

        if user is not None:
            login(request, user)
            messages.success(request, "You are now logged in.")

            if user.is_staff or user.is_superuser:
                return safe_redirect_url(
                    request,
                    next_url,
                    fallback="admin_dashboard",
                )

            return safe_redirect_url(request, next_url)

        messages.error(request, "Invalid username/email or password.")

    return render(
        request,
        "recommender/login.html",
        {"next": next_url},
    )


# ---------------------------------------------------------
# USER LOGOUT
# ---------------------------------------------------------

def logout_view(request):
    if request.method in ("GET", "POST"):
        logout(request)
        messages.success(request, "You have been logged out.")

    return redirect("login")


# ---------------------------------------------------------
# CROP PREDICTION
# ---------------------------------------------------------

@login_required(login_url="login")
def predict_view(request):
    """Validate inputs, predict a crop, and save valid predictions."""

    if request.user.is_staff or request.user.is_superuser:
        return redirect("admin_dashboard")

    weather_info = None
    error_message = None
    inputs = {}
    prediction = None
    city = ""

    if request.method == "POST":
        city = request.POST.get("city", "").strip()
        use_live_weather = (
            request.POST.get("use_live_weather") == "on"
        )

        try:
            # Read the required measurements.
            N = parse_number(request.POST, "N")
            P = parse_number(request.POST, "P")
            K = parse_number(request.POST, "K")
            ph = parse_number(request.POST, "ph")

            temperature = parse_number(request.POST, "temperature")
            humidity = parse_number(request.POST, "humidity")
            rainfall = parse_number(request.POST, "rainfall")

            # Use live weather only when valid data is available.
            if use_live_weather:
                if not city:
                    messages.warning(
                        request,
                        "Enter a city to use live weather. "
                        "Manual values will be used.",
                    )
                else:
                    try:
                        lat, lon, resolved_city = get_coordinates(city)

                        if lat is not None and lon is not None:
                            live_data = fetch_climate_data(lat, lon)

                            if live_data:
                                weather_values = {
                                    "temperature": float(
                                        live_data["temperature"]
                                    ),
                                    "humidity": float(
                                        live_data["humidity"]
                                    ),
                                    "rainfall": float(
                                        live_data["rainfall"]
                                    ),
                                }

                                if not all(
                                    math.isfinite(value)
                                    for value in weather_values.values()
                                ):
                                    raise ValueError(
                                        "Weather data contains invalid values."
                                    )

                                temperature = weather_values["temperature"]
                                humidity = weather_values["humidity"]
                                rainfall = weather_values["rainfall"]

                                weather_info = (
                                    f"Live weather fetched for {resolved_city}: "
                                    f"{temperature}°C, {humidity}% humidity, "
                                    f"{rainfall} mm rainfall."
                                )
                            else:
                                messages.warning(
                                    request,
                                    "Live weather is unavailable. "
                                    "Manual climate values will be used.",
                                )
                        else:
                            messages.warning(
                                request,
                                "City not found. "
                                "Manual climate values will be used.",
                            )

                    except Exception:
                        logger.exception("Live weather lookup failed.")
                        messages.warning(
                            request,
                            "Could not retrieve live weather. "
                            "Manual climate values will be used.",
                        )

            inputs = {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }

            # Validate before prediction or saving.
            validate_crop_inputs(inputs)

            bundle = get_prediction_bundle()

            if bundle is None:
                raise RuntimeError(
                    "The trained crop prediction model is unavailable."
                )

            prediction = predict_one(bundle, inputs)

            if (
                not isinstance(prediction, str)
                or not prediction.strip()
            ):
                prediction = None
                raise RuntimeError(
                    "The model did not return a valid crop recommendation."
                )

            # Save only after validation and successful prediction.
            save_prediction_if_logged_in(
                request,
                inputs,
                prediction,
            )

            messages.success(
                request,
                f"Crop prediction completed: {prediction}.",
            )

        except ValueError as exc:
            error_message = str(exc)

        except Exception:
            logger.exception("Crop prediction failed.")
            prediction = None
            error_message = (
                "The crop prediction could not be completed. "
                "Please check your inputs or try again later."
            )

    return render(
        request,
        "recommender/predict.html",
        {
            "prediction": prediction,
            "weather_info": weather_info,
            "error_message": error_message,
            "inputs": inputs,
            "city": city,
        },
    )


# ---------------------------------------------------------
# USER PREDICTION HISTORY
# ---------------------------------------------------------

@login_required(login_url="login")
def user_history_view(request):
    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "recommender/user_history.html",
        {
            "predictions": predictions,
            "history": predictions,
        },
    )


@login_required(login_url="login")
@require_POST
def user_delete_prediction(request, id):
    prediction = get_object_or_404(
        Prediction,
        id=id,
        user=request.user,
    )

    prediction.delete()
    messages.success(request, "Prediction deleted successfully.")

    return redirect("user_history")


# ---------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------

@login_required(login_url="login")
def profile_view(request):
    profile, _ = UserProfile.objects.get_or_create(
        user=request.user
    )

    if request.method == "POST":
        phone = request.POST.get("phone", "").strip()

        if len(phone) > 20:
            messages.error(
                request,
                "Phone number must be 20 characters or fewer.",
            )
        else:
            profile.phone = phone
            profile.save(update_fields=["phone"])
            messages.success(
                request,
                "Profile updated successfully.",
            )
            return redirect("profile")

    return render(
        request,
        "recommender/profile.html",
        {
            "profile": profile,
            "user_profile": profile,
        },
    )


# ---------------------------------------------------------
# CHANGE PASSWORD
# ---------------------------------------------------------

@login_required(login_url="login")
def change_password_view(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)

        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)

            messages.success(
                request,
                "Your password was changed successfully.",
            )
            return redirect("change_password")
    else:
        form = PasswordChangeForm(request.user)

    return render(
        request,
        "recommender/change_password.html",
        {"form": form},
    )


# ---------------------------------------------------------
# CUSTOM ADMIN LOGIN
# ---------------------------------------------------------

def admin_login_view(request):
    if request.user.is_authenticated and (
        request.user.is_staff or request.user.is_superuser
    ):
        return redirect("admin_dashboard")

    next_url = request.POST.get("next") or request.GET.get("next", "")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        user = authenticate(
            request,
            username=username,
            password=password,
        )

        if user is None and username:
            matching_user = User.objects.filter(
                email__iexact=username
            ).first()

            if matching_user:
                user = authenticate(
                    request,
                    username=matching_user.username,
                    password=password,
                )

        if user is not None and (
            user.is_staff or user.is_superuser
        ):
            login(request, user)
            messages.success(request, "Admin login successful.")

            return safe_redirect_url(
                request,
                next_url,
                fallback="admin_dashboard",
            )

        messages.error(
            request,
            "Invalid credentials or insufficient permissions.",
        )

    return render(
        request,
        "recommender/admin_login.html",
        {"next": next_url},
    )


# ---------------------------------------------------------
# ADMIN DASHBOARD
# ---------------------------------------------------------

@user_passes_test(is_admin, login_url="admin_login")
def admin_dashboard_view(request):
    today = timezone.localdate()

    total_users = User.objects.filter(
        is_staff=False,
        is_superuser=False,
    ).count()

    total_predictions = Prediction.objects.count()

    predictions_today = Prediction.objects.filter(
        created_at__date=today,
    ).count()

    crop_stats = list(
        Prediction.objects.values("predicted_label")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    recent_predictions = Prediction.objects.select_related(
        "user"
    ).order_by("-created_at")[:10]

    context = {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "crop_stats": crop_stats,
        "recent_predictions": recent_predictions,
    }

    return render(
        request,
        "recommender/admin_dashboard.html",
        context,
    )


# ---------------------------------------------------------
# ADMIN: VIEW USERS
# ---------------------------------------------------------

@user_passes_test(is_admin, login_url="admin_login")
def admin_view_users_view(request):
    users = User.objects.filter(
        is_superuser=False,
        is_staff=False,
    ).order_by("username")

    return render(
        request,
        "recommender/admin_view_users.html",
        {"users": users},
    )


# ---------------------------------------------------------
# ADMIN: DELETE INACTIVE USER
# ---------------------------------------------------------

@user_passes_test(is_admin, login_url="admin_login")
@require_POST
def delete_inactive_user_view(request, user_id):
    user = get_object_or_404(
        User.objects.filter(
            is_active=False,
            is_staff=False,
            is_superuser=False,
        ),
        id=user_id,
    )

    user.delete()

    messages.success(
        request,
        "Inactive user deleted successfully.",
    )
    return redirect("admin_view_users")
