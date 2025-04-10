import os

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

print( os.getcwd() )
df = pd.read_csv("./data/Titanic.csv")
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

model = joblib.load('model_v1.pkl')
y_pred = model.predict(X)

print(f"Accuracy (Model V1): {accuracy_score(y, y_pred)}")
