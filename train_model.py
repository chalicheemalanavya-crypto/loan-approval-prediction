import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("loan.csv")

# Features and target
X = data[["Income", "CreditScore", "LoanAmount", "Age"]]
y = data["Approved"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")