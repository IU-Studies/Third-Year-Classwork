import pandas as pd
import numpy as np

# 1. Loading the Titanic Dataset
df = pd.read_csv(r"C:\Users\mayur\Downloads\titanic.csv")

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




# Importing required libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# 1. Load and Explore the Dataset 
titanic = pd.read_csv(r'C:\Users\mayur\Downloads\titanic.csv') 

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





# 1. Event Dependence and Independence 
import numpy as np 

# Simulate two dice rolls 
A = np.random.randint(1, 7, 1000) 
B = np.random.randint(1, 7, 1000) 

# Compute probabilities 
P_A = np.mean(A > 3) 
P_B = np.mean(B % 2 == 0) 
P_A_and_B = np.mean((A > 3) & (B % 2 == 0)) 
P_A_given_B = P_A_and_B / P_B 

print("P(A):", P_A) 
print("P(B):", P_B) 
print("P(A∩B):", P_A_and_B) 
print("P(A|B):", P_A_given_B) 

# Check independence: P(A∩B) should equal P(A)*P(B) 
independent = np.isclose(P_A_and_B, P_A * P_B) 
print("Are A and B independent?", independent) 

# 2. Conditional Probability using a Contingency Table 
import pandas as pd 

# Create contingency table 
data = { 'Passed Math': [30, 20], 'Failed Math': [10, 40]} 
table = pd.DataFrame(data, index=['Passed English', 'Failed English']) 
print(table) 

# P(Passed Math | Passed English) 
P_PM_PE = table.loc['Passed English', 'Passed Math'] / table.loc['Passed English'].sum() 
print("P(Passed Math | Passed English):", P_PM_PE) 

# 3. Bayes’s Theorem Example 
# Given values 
P_spam = 0.01 
P_not_spam = 0.99 
P_positive_given_spam = 0.99 
P_positive_given_not_spam = 0.05 

# Total probability of positive test 
P_positive = (P_positive_given_spam * P_spam) + (P_positive_given_not_spam * P_not_spam) 

# Bayes’ theorem 
P_spam_given_positive = (P_positive_given_spam * P_spam) / P_positive 
print("P(Spam | Positive):", P_spam_given_positive) 

# 4. Random Variables & Continuous Distributions 
# Generate random values from normal distribution 
data = np.random.normal(loc=50, scale=10, size=1000) 

# Mean, Std Dev 
mean = np.mean(data) 
std_dev = np.std(data) 

# Probability P(40 < X < 60) 
prob = np.mean((data > 40) & (data < 60)) 
print("Mean:", mean) 
print("Standard Deviation:", std_dev) 
print("P(40 < X < 60):", prob) 

# 5. Central Limit Theorem (CLT) Simulation 
# Simulate exponential distribution 
population = np.random.exponential(scale=2, size=10000) 

# Sample means 
sample_means = [np.mean(np.random.choice(population, 30)) for _ in range(1000)] 

# Summary statistics 
mean_sample_means = np.mean(sample_means) 
std_sample_means = np.std(sample_means) 
print("Mean of Sample Means:", mean_sample_means) 
print("Standard Deviation of Sample Means:", std_sample_means)




import pandas as pd  # for data manipulation and analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
tips = pd.read_csv(r"C:\Users\mayur\Downloads\tips.csv")

# Univariate Analysis
# Histogram
sns.histplot(tips['total_bill'], kde=True)
plt.title('Histogram of Total Bill')
plt.show()

# Boxplot
sns.boxplot(x=tips['total_bill'])
plt.title('Boxplot of Total Bill')
plt.show()

# Bivariate Analysis
# Scatter Plot
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Scatter Plot of Total Bill vs Tip')
plt.show()

# Boxplot by Category
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Boxplot of Total Bill by Day')
plt.show()

# Multivariate Analysis
# Pair Plot with Hue
sns.pairplot(tips, hue='sex')
plt.show()

# Heatmap of Correlations
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Facet Grid
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()




