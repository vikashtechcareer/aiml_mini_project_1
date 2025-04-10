from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
import joblib

app = Flask(__name__)

# Load the dataset
iris = load_iris()

# Train a Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(iris.data, iris.target)

# Save the model
joblib.dump(model, 'model.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['data']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
