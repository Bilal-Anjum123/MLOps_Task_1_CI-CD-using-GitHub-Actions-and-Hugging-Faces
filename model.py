# File: model.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Use only 'MedInc' (median income) to predict 'MedHouseVal' (median house value)
X = df[['MedInc']]
y = df['MedHouseVal']

# Reduce the dataset size to 1000 records for simplicity
X, y = X[:1000], y[:1000]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
