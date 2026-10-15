import os
import django

def pytest_configure():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "crop_recommendation_site.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()