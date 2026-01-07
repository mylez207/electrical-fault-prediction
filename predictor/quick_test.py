import requests
import json

url = "http://127.0.0.1:8000/api/predict/"

data = {
    "humidity": 50,
    "rainfall": 10,
    "lightning": 0,
    "temperature": 30,
    "wind_speed": 5,
    "weather_severity": 1,
    "voltage_unbalance": 0.5,
    "current_unbalance": 0.5,
    "power_factor": 0.95,
    "frequency": 50,
    "line_loading": 70,
    "active_power": 100,
    "reactive_power": 50,
    "equipment_age": 5,
    "thermal_stress": 0.3,
    "risk_score": 0.2
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    print(f"Response text: {response.text}")