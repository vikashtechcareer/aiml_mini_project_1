from datetime import datetime

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model_v1.pkl')
MODEL_VERSION = "v1"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]  # expecting dict with 'Pclass', 'Sex', 'Age'
    df = pd.DataFrame([data])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    prediction = model.predict(df)

    # Log the request
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": data,
        "prediction": int(prediction),
        "model_version": MODEL_VERSION
    }
    # log into some log file
    with open("predictions.log", "a") as f:
        f.write(f"{log_entry}\n")

    print(f"[LOG] {log_entry}")

    return jsonify({
        "prediction": int(prediction),
        "model_version": MODEL_VERSION
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
