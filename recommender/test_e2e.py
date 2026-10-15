import pytest
from playwright.sync_api import Page, expect
from django.contrib.auth.models import User

@pytest.mark.django_db(transaction=True)
def test_full_user_flow(page: Page, live_server):
    # 1. Create user in test DB
    User.objects.create_user(username="e2e_flow_user@example.com", email="e2e_flow_user@example.com", password="SecurePassword123!")
    
    # 2. Log in via UI to establish live session
    page.goto(f"{live_server.url}/login/")
    page.fill("input[name='email']", "e2e_flow_user@example.com")
    page.fill("input[name='password']", "SecurePassword123!")
    page.click("button[type='submit']")
    expect(page).to_have_url(f"{live_server.url}/predict/")
    
    # 3. Fill prediction inputs
    page.fill("input[name='N']", "90")
    page.fill("input[name='P']", "42")
    page.fill("input[name='K']", "43")
    page.fill("input[name='temperature']", "20.8")
    page.fill("input[name='humidity']", "82.0")
    page.fill("input[name='ph']", "6.5")
    page.fill("input[name='rainfall']", "202.9")
    
    # 4. Target form submit button specifically by ID
    expect(page.locator("#predictButton")).to_be_visible()

@pytest.mark.django_db(transaction=True)
def test_admin_dashboard_access_denied_for_regular_user(page: Page, live_server):
    User.objects.create_user(username="regular_user@example.com", email="regular_user@example.com", password="Password123!")
    page.goto(f"{live_server.url}/login/")
    page.fill("input[name='email']", "regular_user@example.com")
    page.fill("input[name='password']", "Password123!")
    page.click("button[type='submit']")
    page.goto(f"{live_server.url}/admin_dashboard/")
    expect(page).to_have_url(f"{live_server.url}/admin_login/?next=/admin_dashboard/")
