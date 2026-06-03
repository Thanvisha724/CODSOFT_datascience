# Titanic Survival Prediction
# CodSoft Data Science Internship - Task 1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Load Dataset
data = pd.read_csv("titanic.csv")
# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Handle Missing Values
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

# Convert Gender into Numerical Values
data["Sex"] = data["Sex"].map({
    "male": 0,
    "female": 1
})

# Select Features and Target
X = data[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]]
y = data["Survived"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)

# Print Results
print("\n==============================")
print("TITANIC SURVIVAL PREDICTION")
print("==============================")
print(f"Model Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Sample Prediction
sample_passenger = [[1, 1, 25, 100, 0, 0]]

result = model.predict(sample_passenger)

print("\nSample Passenger Prediction:")
if result[0] == 1:
    print("Passenger is likely to SURVIVE")
else:
    print("Passenger is NOT likely to SURVIVE")