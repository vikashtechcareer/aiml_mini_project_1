import joblib
from sklearn.ensemble import RandomForestClassifier

from aiml_mini_project_1.titanic_ab_testing.model_v1.train import X_train, y_train

# same preprocessing as model_v1

# Just change the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model_v2.pkl')
