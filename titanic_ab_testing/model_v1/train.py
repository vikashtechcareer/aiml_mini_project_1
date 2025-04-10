# Logistic Regression model
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

print( os.getcwd() )
df = pd.read_csv('titanic_ab_testing/data/Titanic.csv')
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model_v1.pkl')
