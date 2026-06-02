import pandas as pd
import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

print("TRAINING MODEL TITANIC")
print("="*40)

# Load data
X_train = pd.read_csv("titanic_preprocessed/X_train.csv")
X_test = pd.read_csv("titanic_preprocessed/X_test.csv")
y_train = pd.read_csv("titanic_preprocessed/y_train.csv").values.ravel()
y_test = pd.read_csv("titanic_preprocessed/y_test.csv").values.ravel()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Train model
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Save model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training selesai!")