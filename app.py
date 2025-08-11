import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
reg_model = pickle.load(open("lin_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict.api', methods=['POST'])
def predict_api():
    json_data = request.get_json()

    if not json_data:
        return {"error": "No JSON received. Send valid JSON."}, 400

    data = json_data.get('data')
    if not data:
        return {"error": "Missing 'data' key in JSON."}, 400

    try:
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        output = reg_model.predict(new_data)[0]
        return {"prediction": float(output)}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)
#F:\House_price\Boston\demo\House_price_prediction_model\app.py