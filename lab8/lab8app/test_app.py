# lab8app/test_app.py

import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Example input (match your model's expected features)
data = {
    "feature1": 1.2,
    "feature2": 3.4
}

# Send POST request
response = requests.post(url, json=data)

# Print the prediction
print("Prediction:", response.json())