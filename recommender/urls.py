from django.urls import path

from .views import (
    home,
    signup_view,
    predict_view,
    logout_view,
    login_view,
    user_history_view,
    user_delete_prediction,
    profile_view,
    change_password_view,
    admin_login_view,
    admin_dashboard_view,
    admin_view_users_view,
    delete_inactive_user_view
)


urlpatterns = [
    # Home page
    path("", home, name="home"),

    # User authentication
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),

    # Crop prediction
    path("predict/", predict_view, name="predict"),

    # User prediction history
    path("user_history/", user_history_view, name="user_history"),
    path(
        "history_delete/<int:id>/",
        user_delete_prediction,
        name="user_delete_prediction",
    ),

    # User profile and password
    path("profile/", profile_view, name="profile"),
    path(
        "change_password/",
        change_password_view,
        name="change_password",
    ),

    # Custom admin pages
    path("admin_login/", admin_login_view, name="admin_login"),
    path(
        "admin_dashboard/",
        admin_dashboard_view,
        name="admin_dashboard",
    ),
    path(
        "admin_view_users/",
        admin_view_users_view,
        name="admin_view_users",
    ),
    path(
        "delete_inactive_user/<int:user_id>/",
        delete_inactive_user_view,
        name="delete_inactive_user",
    ),
]
