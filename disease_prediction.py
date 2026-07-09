# Disease Prediction using Machine Learning

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("heart.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# Features and Target
# ==========================

X = df.drop("target", axis=1)
y = df["target"]

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Standardize Data
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Logistic Regression
# ==========================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print("Accuracy:", round(lr_accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, lr_prediction))

print("Confusion Matrix")
print(confusion_matrix(y_test, lr_prediction))

# ==========================
# Support Vector Machine
# ==========================

print("\n==============================")
print("SUPPORT VECTOR MACHINE")
print("==============================")

svm = SVC()

svm.fit(X_train, y_train)

svm_prediction = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_prediction)

print("Accuracy:", round(svm_accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, svm_prediction))

print("Confusion Matrix")
print(confusion_matrix(y_test, svm_prediction))

# ==========================
# Random Forest
# ==========================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print("Accuracy:", round(rf_accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, rf_prediction))

print("Confusion Matrix")
print(confusion_matrix(y_test, rf_prediction))

# ==========================
# Model Comparison
# ==========================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"Logistic Regression : {lr_accuracy*100:.2f}%")
print(f"SVM                 : {svm_accuracy*100:.2f}%")
print(f"Random Forest       : {rf_accuracy*100:.2f}%")

# ==========================
# New Patient Prediction
# ==========================

print("\n==============================")
print("NEW PATIENT PREDICTION")
print("==============================")

# Example patient
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,
# exang,oldpeak,slope,ca,thal

patient = np.array([
    [52, 1, 2, 130, 250, 0, 1, 170, 0, 1.2, 2, 0, 2]
])

patient = scaler.transform(patient)

prediction = rf.predict(patient)

if prediction[0] == 1:
    print("\nPrediction: Heart Disease Detected")
else:
    print("\nPrediction: No Heart Disease")

print("\nProject Executed Successfully!")