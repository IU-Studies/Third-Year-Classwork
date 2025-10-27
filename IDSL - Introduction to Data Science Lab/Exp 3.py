# Importing required libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# 1. Load and Explore the Dataset 
titanic = pd.read_csv(r'C:\Users\IU\Downloads\titanic.csv') 

# Basic information 
print("Shape:", titanic.shape) 
print("Columns:", titanic.columns) 
print(titanic.dtypes) 
print(titanic.head()) 

# 2. Central Tendency 
print("Age - Mean:", titanic['Age'].mean()) 
print("Age - Median:", titanic['Age'].median()) 
print("Age - Mode:", titanic['Age'].mode().values) 

print("Fare - Mean:", titanic['Fare'].mean()) 
print("Fare - Median:", titanic['Fare'].median()) 
print("Fare - Mode:", titanic['Fare'].mode().values) 

# 3. Dispersion of Data 
# Range 
print("Age Range:", titanic['Age'].max() - titanic['Age'].min()) 
print("Fare Range:", titanic['Fare'].max() - titanic['Fare'].min()) 

# Five-number summary 
print(titanic[['Age', 'Fare']].describe()) 

# Variance & Standard Deviation 
print("Age Variance:", titanic['Age'].var()) 
print("Fare Variance:", titanic['Fare'].var()) 
print("Age Std Dev:", titanic['Age'].std()) 
print("Fare Std Dev:", titanic['Fare'].std()) 

# IQR 
iqr_age = titanic['Age'].quantile(0.75) - titanic['Age'].quantile(0.25) 
iqr_fare = titanic['Fare'].quantile(0.75) - titanic['Fare'].quantile(0.25) 
print("Age IQR:", iqr_age) 
print("Fare IQR:", iqr_fare) 

# 4. Correlation 
print("\nCorrelation between Age and Fare:")
print(titanic[['Age', 'Fare']].corr()) 

# 5. Simpson’s Paradox Example 
# Overall survival rate 
print("Overall survival rate:", titanic['Survived'].mean()) 

# Survival rate by gender 
print("\nSurvival rate by gender:")
print(titanic.groupby('Sex')['Survived'].mean()) 

# Survival rate by class and gender 
print("\nSurvival rate by class and gender:")
print(titanic.groupby(['Pclass', 'Sex'])['Survived'].mean()) 

# 6. Visualization 
sns.barplot(x='Sex', y='Survived', data=titanic) 
plt.title('Survival Rate by Gender') 
plt.show() 

sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=titanic) 
plt.title('Survival by Class and Gender') 
plt.show()
