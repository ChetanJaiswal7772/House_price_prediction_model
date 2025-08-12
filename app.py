import pickle
import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
reg_model = pickle.load(open(os.path.join(BASE_DIR, "lin_reg_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# Column names and friendly names
columns = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

friendly_names = {
    "MedInc": "Median Income",
    "HouseAge": "House Age",
    "AveRooms": "Average Rooms",
    "AveBedrms": "Average Bedrooms",
    "Population": "Population",
    "AveOccup": "Average Occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

@app.route('/')
def home():
    # Default form values
    default_values = {col: "" for col in columns}
    return render_template(
        "home.html",
        columns=default_values,
        friendly_names=friendly_names,
        prediction_text=""
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values in the correct order
        input_data = [float(request.form[col]) for col in columns]
        final_input = scaler.transform([input_data])
        prediction = reg_model.predict(final_input)[0]

        values_entered = {col: request.form[col] for col in columns}

        return render_template(
            "home.html",
            columns=values_entered,
            friendly_names=friendly_names,
            prediction_text=f"Predicted House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        values_entered = {col: request.form.get(col, "") for col in columns}
        return render_template(
            "home.html",
            columns=values_entered,
            friendly_names=friendly_names,
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    # In production, debug should be False
    app.run(host="0.0.0.0", port=5000, debug=False)
