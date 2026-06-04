import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/churn_data.csv")

# Features and target
X = df[["Age", "MonthlyCharges", "Tenure"]]
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nCustomer Churn Prediction")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Results
result = X_test.copy()
result["Actual"] = y_test.values
result["Predicted"] = predictions

print("\nPrediction Results")
print(result)

# Graph
churn_count = df["Churn"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(["No Churn", "Churn"], churn_count)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.show()