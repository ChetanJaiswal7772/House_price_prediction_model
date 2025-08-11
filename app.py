import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
reg_model = pickle.load(open("lin_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Column names with empty values (you can prefill if you want defaults)
columns = {
    "MedInc": "",
    "HouseAge": "",
    "AveRooms": "",
    "AveBedrms": "",
    "Population": "",
    "AveOccup": "",
    "Latitude": "",
    "Longitude": ""
}

@app.route('/')
def home():
    return render_template("home.html", columns=columns)

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

@app.route('/predict', methods=['POST'])
def predict():
    feature_order = list(columns.keys())
    try:
        data = [float(request.form[col]) for col in feature_order]

        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = reg_model.predict(final_input)[0]

        updated_cols = dict(zip(feature_order, request.form.values()))
        return render_template(
            "home.html",
            columns=updated_cols,
            prediction_text=f"The house price prediction is {output:.2f}"
        )
    except Exception as e:
        return render_template(
            "home.html",
            columns=columns,
            prediction_text=f"Error: {e}"
        )

if __name__ == "__main__":
    app.run(debug=True)
