import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


def test_home():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_prediction():
    client = app.test_client()

    data = {
        "City": "Ahmedabad",
        "PM2.5": 50,
        "PM10": 80,
        "NO": 20,
        "NO2": 30,
        "NOx": 40,
        "NH3": 10,
        "CO": 1,
        "SO2": 15,
        "O3": 50,
        "Benzene": 2,
        "Toluene": 5,
        "Xylene": 1
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.get_json()

    assert "predicted_AQI" in result
    assert isinstance(result["predicted_AQI"], (int, float))