from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)
joblib.dump(model, 'model.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['data']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
