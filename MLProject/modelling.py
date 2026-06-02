import pandas as pd
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("TRAINING MODEL TITANIC")
print("="*50)


X_train = pd.read_csv("titanic_preprocessed/X_train.csv")
X_test = pd.read_csv("titanic_preprocessed/X_test.csv")
y_train = pd.read_csv("titanic_preprocessed/y_train.csv").values.ravel()
y_test = pd.read_csv("titanic_preprocessed/y_test.csv").values.ravel()

print(f"Data: X_train={X_train.shape}, X_test={X_test.shape}")


model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")


with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as best_model.pkl")
print("TRAINING SELESAI")