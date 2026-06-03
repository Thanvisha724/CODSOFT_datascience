import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading Dataset...")

# Load Dataset
data = pd.read_csv("salesprediction/advertising.csv")

print("\nDataset Preview:")
print(data.head())

# Features and Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
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
sample = [[230.1, 37.8, 69.2]]
predicted_sales = model.predict(sample)

print("\nPredicted Sales:")
print(round(predicted_sales[0], 2))