import pandas as pd
import numpy as np

# 1. Loading the Titanic Dataset
df = pd.read_csv(r"C:\Users\IU\Downloads\titanic.csv")

# Printing first 5 rows of dataset
print("The first 5 rows of titanic dataset:")
print(df.head())

# 2. Checking dataset info
print("Data types along with non-null counts:")
print(df.info())

# Measurement Scale-based Classification (manual classification)
print("\nMeasurement Scale Classification:")
# Create a dictionary with classifications
measurement_scale = {
    'PassengerId': 'Nominal',
    'Survived': 'Nominal',
    'Pclass': 'Ordinal',
    'Name': 'Nominal',
    'Sex': 'Nominal',
    'Age': 'Ratio',
    'SibSp': 'Ratio',
    'Parch': 'Ratio',
    'Ticket': 'Nominal',
    'Fare': 'Ratio',
    'Cabin': 'Nominal',
    'Embarked': 'Nominal'
}

# Create a DataFrame showing column name, data type, and measurement scale
column_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type (Pandas)': df.dtypes.values,
    'Measurement Scale': [measurement_scale[col] for col in df.columns]
})

# Printing the dataset
print(column_info)

# 3. Performing Data Preprocessing
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to many missing values
df.drop('Cabin', axis=1, inplace=True)

# Convert categorical columns to category type
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Printing data after preprocessing
print("\nDataset after preprocessing:")
print(df.head())

# Last check for missing values
print("\nRemaining missing values after preprocessing:")
print(df.isnull().sum())

# Printing updated data types
print("\nUpdated Data Types after cleaning:")
print(df.dtypes)
