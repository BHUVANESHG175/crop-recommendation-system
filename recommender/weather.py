import requests
from datetime import datetime, timedelta

def get_coordinates(city_name):
    """Convert city name to latitude/longitude using Open-Meteo API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and res.json().get("results"):
            loc = res.json()["results"][0]
            return loc["latitude"], loc["longitude"], loc.get("name")
    except Exception:
        pass
    return None, None, None

def fetch_climate_data(lat, lon):
    """Fetch live temp/humidity and 30-day cumulative rainfall."""
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=30)
    current_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={thirty_days_ago}&end_date={today}&daily=precipitation_sum"
    try:
        c_res = requests.get(current_url, timeout=5).json()
        a_res = requests.get(archive_url, timeout=5).json()
        temp = c_res["current"]["temperature_2m"]
        hum = c_res["current"]["relative_humidity_2m"]
        precip = sum([p for p in a_res.get("daily", {}).get("precipitation_sum", []) if p is not None])
        return {"temperature": round(temp, 1), "humidity": round(hum, 1), "rainfall": round(precip, 1)}
    except Exception:
        return None