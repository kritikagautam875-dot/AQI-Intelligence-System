from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# Load trained AQI pipeline
model = joblib.load("model/final_aqi_rf_pipeline.pkl")


app = FastAPI(
    title="AQI Prediction API",
    description="FastAPI service for predicting Air Quality Index",
    version="1.0.0"
)


# Input schema
class AQIInput(BaseModel):
    City: str

    PM2_5: float = Field(alias="PM2.5")
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float

    class Config:
        populate_by_name = True


@app.get("/")
def home():
    return {
        "message": "AQI Prediction FastAPI is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_aqi(data: AQIInput):

    input_data = pd.DataFrame([{
        "City": data.City,
        "PM2.5": data.PM2_5,
        "PM10": data.PM10,
        "NO": data.NO,
        "NO2": data.NO2,
        "NOx": data.NOx,
        "NH3": data.NH3,
        "CO": data.CO,
        "SO2": data.SO2,
        "O3": data.O3,
        "Benzene": data.Benzene,
        "Toluene": data.Toluene,
        "Xylene": data.Xylene
    }])

    prediction = model.predict(input_data)[0]

    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    return {
        "predicted_AQI": round(prediction, 2),
        "category": category
    }