# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv("data.csv")

# Show the first few rows of the dataset
print("First 5 rows of the dataset:")
print(dataset.head())

# Separate features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("\nSample features before imputation:")
print(X[:5])

# Handling missing data (Impute columns 1 and 2 - make sure these columns exist in your dataset)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("\nSample features after imputation:")
print(X[:5])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print("\nSample features after encoding categorical data:")
print(X[:5])

print("\nSample target labels after encoding:")
print(y[:5])

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"\nSize of training set: {X_train.shape}")
print(f"Size of test set: {X_test.shape}")
