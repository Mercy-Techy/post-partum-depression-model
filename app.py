import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load  trained Logistic regression model 
model = joblib.load("logistic_postpartum_depression.pkl")

@app.route("/", methods=["GET"])
def run():
    return "Flask API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', None)

    if symptoms is None or len(symptoms) != 10:
        return jsonify({'error': 'Please fill all the input boxes'}), 400

    try:
        symptoms = np.array(symptoms, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400

    prediction = model.predict(symptoms)[0]
    proba = model.predict_proba(symptoms)[0][1]
    print(prediction,proba,symptoms)
    if prediction == 1:
        result = "Based on your responses, there are signs that may indicate postpartum depression. We recommend seeking support from a healthcare provider or a mental health professional for further guidance." 
    else:
        result = "Your responses do not currently suggest signs of postpartum depression. However, if you're still feeling overwhelmed or unsure, don’t hesitate to reach out for support."

    return jsonify({
        "prediction": result,
        "probability": round(float(proba), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
