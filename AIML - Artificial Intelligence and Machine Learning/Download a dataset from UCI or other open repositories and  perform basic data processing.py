"""
Download a dataset from UCI or other open repositories and 
perform basic data processing using python /R including 
handling missing values, encoding and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("First 5 rows of dataset:\n", df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Handling:\n", df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nData after Encoding:\n", df.head())

# Normalize numeric columns
numeric_cols = ['Age', 'Fare']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nData after Normalization:\n", df[numeric_cols].head())

# Final processed dataset
print("\nFinal Processed Dataset:\n", df.head())

# Save preprocessed data
df.to_csv("titanic_preprocessed.csv", index=False)
