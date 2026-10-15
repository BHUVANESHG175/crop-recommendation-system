from unittest.mock import patch
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from recommender.models import Prediction, UserProfile

class SignupViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.signup_url = reverse("signup")

    def test_signup_successful(self):
        response = self.client.post(self.signup_url, {
            "full_name": "Jane Doe",
            "phone_number": "+1234567890",
            "email": "jane@example.com",
            "password": "StrongPassword123!",
            "confirm_password": "StrongPassword123!",
            "agree": "yes",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(email="jane@example.com").exists())

class PredictViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="user@example.com",
            email="user@example.com",
            password="Password123!"
        )
        self.staff_user = User.objects.create_user(
            username="admin@example.com",
            email="admin@example.com",
            password="Password123!",
            is_staff=True
        )
        self.predict_url = reverse("predict")

    def test_unauthenticated_user_redirected(self):
        response = self.client.get(self.predict_url)
        self.assertEqual(response.status_code, 302)

    def test_staff_user_redirected_to_admin_dashboard(self):
        self.client.login(username="admin@example.com", password="Password123!")
        response = self.client.get(self.predict_url)
        self.assertRedirects(response, reverse("admin_dashboard"))

    def test_predict_out_of_range_inputs(self):
        self.client.login(username="user@example.com", password="Password123!")
        payload = {
            "N": 250,
            "P": 50,
            "K": 50,
            "temperature": 25.0,
            "humidity": 80.0,
            "ph": 6.5,
            "rainfall": 200.0,
        }
        response = self.client.post(self.predict_url, payload)
        self.assertEqual(Prediction.objects.count(), 0)

    @patch("recommender.views.predict_one")
    @patch("recommender.views.load_bundle")
    def test_predict_success_creates_database_record(self, mock_load, mock_predict):
        mock_load.return_value = {"features_cols": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]}
        mock_predict.return_value = "rice"

        self.client.login(username="user@example.com", password="Password123!")
        payload = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9,
        }
        response = self.client.post(self.predict_url, payload)
        self.assertEqual(Prediction.objects.count(), 1)
        prediction = Prediction.objects.first()
        self.assertEqual(prediction.predicted_label, "rice")

class AdminDashboardSecurityTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.regular_user = User.objects.create_user(
            username="regular@example.com",
            email="regular@example.com",
            password="Password123!"
        )
        self.admin_url = reverse("admin_dashboard")

    def test_regular_user_cannot_access_admin_dashboard(self):
        self.client.login(username="regular@example.com", password="Password123!")
        response = self.client.get(self.admin_url)
        self.assertNotEqual(response.status_code, 200)
