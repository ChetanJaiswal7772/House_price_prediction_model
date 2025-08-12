
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
# model_path = 'model.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

reg_model = pickle.load(open("lin_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = scaler.predict(final_features)
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('home.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)