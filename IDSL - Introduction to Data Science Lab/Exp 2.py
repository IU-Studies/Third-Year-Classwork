import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 

# 1. Loading the Titanic Dataset 
df = pd.read_csv(r"C:\Users\mayur\Downloads\titanic.csv") 

# Take first few rows 
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Shape of the dataset 
print("\nDataset contains:", df.shape[0], "rows and", df.shape[1], "columns") 

# Info about datatypes and null values 
print("\nDataset info:")
print(df.info())

# 2. Handling Missing Data 
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True) 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) 

# Drop 'Cabin' as it has too many missing values 
df.drop('Cabin', axis=1, inplace=True) 

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. Outlier Detection and Treatment 
plt.figure(figsize=(12, 5)) 

plt.subplot(1, 2, 1) 
sns.boxplot(x=df['Age']) 
plt.title("Boxplot - Age") 

plt.subplot(1, 2, 2) 
sns.boxplot(x=df['Fare']) 
plt.title("Boxplot - Fare") 

plt.tight_layout() 
plt.show() 

# Treating outliers in 'Fare'
Q1 = df['Fare'].quantile(0.25) 
Q3 = df['Fare'].quantile(0.75) 
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR 

df['Fare'] = np.where(
    df['Fare'] > upper_bound, upper_bound,
    np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])
)

# 4. Feature Scaling 
scaler = StandardScaler() 
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']]) 

print("\nScaled values of Age & Fare (first 5 rows):") 
print(df[['Age', 'Fare']].head())

# 5. Encoding Categorical Variables 
print("\nCategorical columns in the dataset:", df.select_dtypes(include='object').columns.tolist()) 

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True) 

# Final print 
print("\nDataset after encoding:") 
print(df.head()) 

print("\nFinal dataset shape:", df.shape)
