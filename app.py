from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained AQI model
model = joblib.load("model/final_aqi_rf_pipeline.pkl")

# Features required by the model
REQUIRED_FEATURES = [
    "City",
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body must contain JSON data"
            }), 400

        # Check for missing features
        missing_features = [
            feature for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing": missing_features
            }), 400

        # Create DataFrame
        input_data = pd.DataFrame([data])

        # Keep only required features
        input_data = input_data[REQUIRED_FEATURES]

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_AQI": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)