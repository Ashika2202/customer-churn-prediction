import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------
# Load Dataset
# --------------------

df = pd.read_csv("data/churn_data.csv")

print("\nDataset Preview")
print(df.head())

# --------------------
# EDA
# --------------------

print("\nDataset Info")
print(df.info())

print("\nStatistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# --------------------
# Features & Target
# --------------------

X = df[['Age', 'MonthlyCharges', 'Tenure']]
y = df['Churn']

# --------------------
# Train/Test Split
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------
# Train Model
# --------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------
# Prediction
# --------------------

y_pred = model.predict(X_test)

# --------------------
# Accuracy
# --------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# --------------------
# Save Predictions
# --------------------

results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

results.to_csv(
    "output/prediction_results.csv",
    index=False
)

# --------------------
# Confusion Matrix
# --------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "output/confusion_matrix.png"
)

plt.show()

print("\nResults saved successfully.")