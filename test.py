import requests
import json

# Step 1: Login to get token
print("Step 1: Logging in...")
login_url = "http://127.0.0.1:8000/api/auth/login/"
login_data = {
    "username": "amina",  # Use your superuser username
    "password": "123456789"  # Use your superuser password
}

login_response = requests.post(login_url, json=login_data)
print(f"Login Status: {login_response.status_code}")

if login_response.status_code != 200:
    print("Login failed!")
    print(login_response.text)
    exit()

token = login_response.json()['token']
print(f"Token received: {token[:20]}...")

# Step 2: Make prediction with token
print("\nStep 2: Making prediction...")
predict_url = "http://127.0.0.1:8000/api/predict/"

headers = {
    "Authorization": f"Token {token}",
    "Content-Type": "application/json"
}

test_data = {
    "humidity": 50.0,
    "rainfall": 10.0,
    "lightning": 0.0,
    "temperature": 30.0,
    "wind_speed": 5.0,
    "weather_severity": 1.0,
    "voltage_unbalance": 0.5,
    "current_unbalance": 0.5,
    "power_factor": 0.95,
    "frequency": 50.0,
    "line_loading": 70.0,
    "active_power": 100.0,
    "reactive_power": 50.0,
    "equipment_age": 5.0,
    "thermal_stress": 0.3,
    "risk_score": 0.2
}

prediction_response = requests.post(predict_url, json=test_data, headers=headers)
print(f"Prediction Status: {prediction_response.status_code}")

if prediction_response.status_code == 200:
    result = prediction_response.json()
    print("\n✓ SUCCESS!")
    print(json.dumps(result, indent=2))
else:
    print("\n✗ FAILED!")
    print(prediction_response.text)