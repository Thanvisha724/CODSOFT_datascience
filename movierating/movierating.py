import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading Dataset...")

# Dataset Load
data = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("\nDataset Shape:", data.shape)

# Missing values remove
data = data.dropna()

# Unwanted column remove
data = data.drop("Name", axis=1)

# Convert text columns to numbers
le = LabelEncoder()

for col in ["Year", "Duration", "Genre", "Director",
            "Actor 1", "Actor 2", "Actor 3", "Votes"]:
    data[col] = le.fit_transform(data[col].astype(str))

# Features and Target
X = data.drop("Rating", axis=1)
y = data["Rating"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== RESULTS ==========")
print("Mean Absolute Error :", round(mae, 2))
print("R2 Score            :", round(r2, 2))

# Sample Prediction
sample = X.iloc[[0]]
predicted_rating = model.predict(sample)

print("\nSample Movie Predicted Rating:")
print(round(predicted_rating[0], 2))