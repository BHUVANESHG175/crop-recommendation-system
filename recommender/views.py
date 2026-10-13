
import json
import logging
import math
import re
from datetime import timedelta

from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import (
    authenticate,
    login,
    logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import IntegrityError, transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .ml.loader import load_bundle, predict_one
from .models import Prediction, UserProfile


logger = logging.getLogger(__name__)

PHONE_PATTERN = re.compile(r"^[0-9+().\-\s]{7,20}$")

EXPECTED_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

def home(request):
    return render(request, "home.html")


# ---------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------

def _valid_phone(phone):
    """Check whether a phone number contains permitted characters."""
    return bool(PHONE_PATTERN.fullmatch(phone))


# ---------------------------------------------------------
# USER SIGNUP
# ---------------------------------------------------------

def signup_view(request):
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect("admin_dashboard")
        return redirect("predict")

    if request.method == "POST":
        name = request.POST.get("full_name", "").strip()
        phone = request.POST.get("phone_number", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")
        accepted_terms = request.POST.get("agree") == "yes"

        if not all([
            name,
            phone,
            email,
            password,
            confirm_password,
        ]):
            messages.error(
                request,
                "Please fill in all required fields.",
            )
            return redirect("signup")

        if not accepted_terms:
            messages.error(
                request,
                "Please accept the terms to create an account.",
            )
            return redirect("signup")

        if len(name) > 150:
            messages.error(request, "Your name is too long.")
            return redirect("signup")

        if not _valid_phone(phone):
            messages.error(
                request,
                "Please enter a valid phone number.",
            )
            return redirect("signup")

        if len(email) > 254:
            messages.error(
                request,
                "Email address is too long.",
            )
            return redirect("signup")

        try:
            validate_email(email)
        except ValidationError:
            messages.error(
                request,
                "Please enter a valid email address.",
            )
            return redirect("signup")

        if password != confirm_password:
            messages.error(
                request,
                "Passwords do not match.",
            )
            return redirect("signup")

        if User.objects.filter(username__iexact=email).exists():
            messages.error(
                request,
                "An account already exists with this email.",
            )
            return redirect("signup")

        if User.objects.filter(email__iexact=email).exists():
            messages.error(
                request,
                "This email address is already registered.",
            )
            return redirect("signup")

        first_name, _, last_name = name.partition(" ")

        candidate_user = User(
            username=email,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )

        try:
            validate_password(password, user=candidate_user)
        except ValidationError as error:
            for error_message in error.messages:
                messages.error(request, error_message)
            return redirect("signup")

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=email,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                )

                UserProfile.objects.create(
                    user=user,
                    phone=phone,
                )

        except IntegrityError:
            messages.error(
                request,
                "An account with these details may already exist. "
                "Please try again.",
            )
            return redirect("signup")

        login(request, user)

        messages.success(
            request,
            "Account created successfully.",
        )
        return redirect("predict")

    return render(request, "signup.html")


# ---------------------------------------------------------
# CROP PREDICTION
# ---------------------------------------------------------

@login_required
def predict_view(request):
    if request.user.is_staff:
        return redirect("admin_dashboard")

    try:
        bundle = load_bundle()
        feature_order = bundle["features_cols"]

        if list(feature_order) != EXPECTED_FEATURES:
            raise ValueError(
                "The model feature order does not match the application."
            )

    except Exception:
        logger.exception(
            "Could not load the crop recommendation model."
        )

        messages.error(
            request,
            "The prediction service is temporarily unavailable. "
            "Please try again later.",
        )

        return render(
            request,
            "predict.html",
            {
                "result": None,
                "last_data": None,
                "feature_order": EXPECTED_FEATURES,
            },
            status=503,
        )

    result = None
    last_data = None

    if request.method == "POST":
        data = {}

        try:
            for column in feature_order:
                raw_value = request.POST.get(
                    column,
                    "",
                ).strip()

                if not raw_value:
                    raise ValueError(
                        f"{column} is required."
                    )

                value = float(raw_value)

                if not math.isfinite(value):
                    raise ValueError(
                        f"{column} must be a finite number."
                    )

                data[column] = value

            if not 0 <= data["N"] <= 200:
                raise ValueError(
                    "Nitrogen (N) must be between 0 and 200."
                )

            if not 0 <= data["P"] <= 200:
                raise ValueError(
                    "Phosphorus (P) must be between 0 and 200."
                )

            if not 0 <= data["K"] <= 250:
                raise ValueError(
                    "Potassium (K) must be between 0 and 250."
                )

            if not -10 <= data["temperature"] <= 60:
                raise ValueError(
                    "Temperature must be between -10 and 60 °C."
                )

            if not 0 <= data["humidity"] <= 100:
                raise ValueError(
                    "Humidity must be between 0 and 100%."
                )

            if not 0 <= data["ph"] <= 14:
                raise ValueError(
                    "Soil pH must be between 0 and 14."
                )

            if not 0 <= data["rainfall"] <= 5000:
                raise ValueError(
                    "Rainfall must be between 0 and 5000 mm."
                )

        except (ValueError, TypeError) as error:
            messages.error(
                request,
                str(error) or
                "Please enter valid numeric values for all fields.",
            )
            return redirect("predict")

        try:
            label = predict_one(data)

            Prediction.objects.create(
                user=request.user,
                predicted_label=str(label),
                **data,
            )

        except Exception:
            logger.exception(
                "Crop prediction failed for user ID %s.",
                request.user.pk,
            )

            messages.error(
                request,
                "We could not generate a prediction. "
                "Please check your inputs and try again.",
            )
            return redirect("predict")

        result = str(label)
        last_data = data

        messages.success(
            request,
            f"Recommended Crop: {result}",
        )

    return render(
        request,
        "predict.html",
        {
            "result": result,
            "last_data": last_data,
            "feature_order": feature_order,
        },
    )


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect("admin_dashboard")
        return redirect("predict")

    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")

        user = authenticate(
            request,
            username=email,
            password=password,
        )

        if user is not None and user.is_active:
            login(request, user)

            messages.success(
                request,
                "Logged in successfully.",
            )

            if user.is_staff:
                return redirect("admin_dashboard")

            return redirect("predict")

        messages.error(
            request,
            "Invalid email or password.",
        )
        return redirect("login")

    return render(request, "login.html")


# ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------

@login_required
@require_POST
def logout_view(request):
    logout(request)

    messages.success(
        request,
        "Logged out successfully.",
    )

    return redirect("login")


# ---------------------------------------------------------
# USER PREDICTION HISTORY
# ---------------------------------------------------------

@login_required
def user_history_view(request):
    if request.user.is_staff:
        return redirect("admin_dashboard")

    predictions = Prediction.objects.filter(
        user=request.user,
    ).order_by("-created_at")

    return render(
        request,
        "history.html",
        {
            "predictions": predictions,
        },
    )


# ---------------------------------------------------------
# DELETE A USER'S OWN PREDICTION
# ---------------------------------------------------------

@login_required
@require_POST
def user_delete_prediction(request, id):
    if request.user.is_staff:
        return redirect("admin_dashboard")

    prediction = get_object_or_404(
        Prediction,
        id=id,
        user=request.user,
    )

    prediction.delete()

    messages.success(
        request,
        "Prediction deleted successfully.",
    )

    return redirect("user_history")


# ---------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------

@login_required
def profile_view(request):
    if request.user.is_staff:
        return redirect("admin_dashboard")

    profile, _ = UserProfile.objects.get_or_create(
        user=request.user,
    )

    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        phone = request.POST.get("phone", "").strip()
        email = request.POST.get("email", "").strip().lower()

        if not name:
            messages.error(
                request,
                "Name cannot be empty.",
            )

        elif len(name) > 150:
            messages.error(
                request,
                "Your name is too long.",
            )

        elif phone and not _valid_phone(phone):
            messages.error(
                request,
                "Please enter a valid phone number.",
            )

        elif not email:
            messages.error(
                request,
                "Email address cannot be empty.",
            )

        elif len(email) > 254:
            messages.error(
                request,
                "Email address is too long.",
            )

        else:
            try:
                validate_email(email)

            except ValidationError:
                messages.error(
                    request,
                    "Please enter a valid email address.",
                )

            else:
                duplicate_email = User.objects.filter(
                    email__iexact=email,
                ).exclude(
                    pk=request.user.pk,
                ).exists()

                duplicate_username = User.objects.filter(
                    username__iexact=email,
                ).exclude(
                    pk=request.user.pk,
                ).exists()

                if duplicate_email or duplicate_username:
                    messages.error(
                        request,
                        "This email address is already associated "
                        "with another account.",
                    )

                else:
                    first_name, _, last_name = name.partition(" ")

                    try:
                        with transaction.atomic():
                            user = User.objects.select_for_update().get(
                                pk=request.user.pk,
                            )

                            user.first_name = first_name
                            user.last_name = last_name
                            user.email = email
                            user.username = email

                            user.save(
                                update_fields=[
                                    "first_name",
                                    "last_name",
                                    "email",
                                    "username",
                                ],
                            )

                            profile.phone = phone

                            profile.save(
                                update_fields=["phone"],
                            )

                    except IntegrityError:
                        messages.error(
                            request,
                            "We could not save your changes because "
                            "the email address may already be in use.",
                        )

                    else:
                        # Keep the current session authenticated.
                        # The username change does not change the password.
                        messages.success(
                            request,
                            "Profile updated successfully.",
                        )

                        return redirect("profile")

    full_name = (
        f"{request.user.first_name} {request.user.last_name}"
    ).strip()

    return render(
        request,
        "profile.html",
        {
            "profile": profile,
            "full_name": full_name,
        },
    )


# ---------------------------------------------------------
# CHANGE PASSWORD
# ---------------------------------------------------------

@login_required
def change_password_view(request):
    if request.method == "POST":
        current = request.POST.get("current_password", "")
        new = request.POST.get("new_password", "")
        confirm = request.POST.get("confirm_password", "")

        if not request.user.check_password(current):
            messages.error(
                request,
                "Incorrect current password.",
            )
            return redirect("change_password")

        if new != confirm:
            messages.error(
                request,
                "New passwords do not match.",
            )
            return redirect("change_password")

        if request.user.check_password(new):
            messages.error(
                request,
                "Your new password must be different from "
                "your current password.",
            )
            return redirect("change_password")

        try:
            validate_password(
                new,
                user=request.user,
            )

        except ValidationError as error:
            for error_message in error.messages:
                messages.error(request, error_message)

            return redirect("change_password")

        request.user.set_password(new)
        request.user.save()

        update_session_auth_hash(
            request,
            request.user,
        )

        messages.success(
            request,
            "Password changed successfully.",
        )

        return redirect("change_password")

    return render(
        request,
        "change_password.html",
    )


# ---------------------------------------------------------
# ADMIN LOGIN
# ---------------------------------------------------------

def admin_login_view(request):
    if request.user.is_authenticated and request.user.is_staff:
        return redirect("admin_dashboard")

    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")

        try:
            user = User.objects.get(
                email__iexact=email,
            )

        except User.DoesNotExist:
            user = None

        if user is not None and user.is_active and user.is_staff:
            authenticated_user = authenticate(
                request,
                username=user.username,
                password=password,
            )

            if authenticated_user is not None:
                login(
                    request,
                    authenticated_user,
                )

                messages.success(
                    request,
                    "Admin login successful.",
                )

                return redirect("admin_dashboard")

        messages.error(
            request,
            "Invalid email or password, or you do not have "
            "admin access.",
        )

    return render(
        request,
        "admin_login.html",
    )


# ---------------------------------------------------------
# ADMIN DASHBOARD
# ---------------------------------------------------------

@staff_member_required(login_url="admin_login")
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
        .annotate(total=Count("id"))
        .order_by("-total")
    )

    most_recommended = (
        crop_stats[0]["predicted_label"]
        if crop_stats
        else "No predictions yet"
    )

    recent_predictions = (
        Prediction.objects.select_related("user")
        .order_by("-created_at")[:10]
    )

    last_7_days = [
        today - timedelta(days=days_ago)
        for days_ago in range(6, -1, -1)
    ]

    daily_stats = (
        Prediction.objects.filter(
            created_at__date__gte=last_7_days[0],
            created_at__date__lte=today,
        )
        .values("created_at__date")
        .annotate(total=Count("id"))
        .order_by("created_at__date")
    )

    stats_by_date = {
        item["created_at__date"]: item["total"]
        for item in daily_stats
    }

    prediction_dates = [
        day.strftime("%b %d")
        for day in last_7_days
    ]

    prediction_counts = [
        stats_by_date.get(day, 0)
        for day in last_7_days
    ]

    context = {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "most_recommended_crop": most_recommended,
        "recent_predictions": recent_predictions,
        "crop_labels": json.dumps(
            [item["predicted_label"] for item in crop_stats]
        ),
        "crop_counts": json.dumps(
            [item["total"] for item in crop_stats]
        ),
        "prediction_dates": json.dumps(prediction_dates),
        "prediction_counts": json.dumps(prediction_counts),
    }

    return render(
        request,
        "admin_dashboard.html",
        context,
    )


# ---------------------------------------------------------
# ADMIN: VIEW AND SEARCH USERS
# ---------------------------------------------------------

@staff_member_required(login_url="admin_login")
def admin_view_users_view(request):
    search_query = request.GET.get("q", "").strip()

    # Only regular user accounts appear in this management page.
    registered_users = User.objects.filter(
        is_staff=False,
        is_superuser=False,
    )

    # Count all regular users, regardless of the search term.
    total_registered_users = registered_users.count()

    users = registered_users.order_by("-date_joined")

    if search_query:
        users = users.filter(
            Q(username__icontains=search_query)
            | Q(email__icontains=search_query)
            | Q(first_name__icontains=search_query)
            | Q(last_name__icontains=search_query)
        )

    return render(
        request,
        "admin_view_users.html",
        {
            "users": users,
            "search_query": search_query,
            "total_registered_users": total_registered_users,
        },
    )


# ---------------------------------------------------------
# ADMIN: DELETE INACTIVE USERS
# ---------------------------------------------------------


@staff_member_required(login_url="admin_login")
@require_POST
def delete_inactive_user_view(request, user_id):
    user = get_object_or_404(
        User,
        pk=user_id,
    )

    # Protect staff and superuser accounts.
    if user.is_staff or user.is_superuser:
        messages.error(
            request,
            "Staff and superuser accounts cannot be deleted here.",
        )
        return redirect("admin_view_users")

    # Prevent an admin from deleting their own account.
    if user.pk == request.user.pk:
        messages.error(
            request,
            "You cannot delete your own account here.",
        )
        return redirect("admin_view_users")

    username = user.username

    try:
        user.delete()
    except Exception:
        logger.exception(
            "Failed to delete user ID %s by admin ID %s.",
            user_id,
            request.user.pk,
        )
        messages.error(
            request,
            "The user could not be deleted. Please try again.",
        )
        return redirect("admin_view_users")

    messages.success(
        request,
        f'User "{username}" was deleted successfully.',
    )

    return redirect("admin_view_users")
