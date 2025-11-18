# Exp1 Implementation of the Data Science Lifecycle and Data Type Classification Using the Titanic Dataset

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




#-----------------------------------------------------------

# Exp2 Perform Exploratory Data Analysis and Data Preprocessing using Python on a Real-World Dataset

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 

# 1. Loading the Titanic Dataset 
df = pd.read_csv(r"C:\Users\IU\Downloads\titanic.csv") 

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



# ---------------------------------------------

# Exp3 To perform statistical analysis on a dataset using Python by computing measures of central tendency, dispersion, correlation, and to illustrate Simpson&#39;s Paradox using suitable visualizations. The Titanic dataset will be used for this analysis.

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



# -------------------------------------------


# Exp4 To apply Python programming for understanding and computing key probability and distribution concepts such as event dependence, conditional probability, Bayes’s theorem, random variables, continuous distributions, and to simulate the Central Limit Theorem (CLT).

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



# -------------------------------------------


# Exp 5 Data visualization technique

import pandas as pd  # for data manipulation and analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
tips = pd.read_csv(r"C:\Users\IU\Downloads\tips.csv")

# Univariate Analysis
# Histogram
sns.histplot(tips['total_bill'])
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
corr = tips.select_dtypes(include=['number']).corr()
sns.heatmap(corr)
plt.title('Correlation Heatmap')
plt.show()


# Facet Grid
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()


# ------------------------------------------

# Exp7 - Implement Dimensionality reduction using Principal Component Analysis method on the dataset Iris.


# Step 1: Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the Digits Dataset
digits = load_digits()
X = digits.data  # Feature matrix (1797 samples x 64 features)
y = digits.target  # Labels (digits 0-9)
print(f"Original Data Shape: {X.shape}")  # (1797, 64)

# Step 3: Normalize the Data (Standardization)
# PCA works best when features are centered and scaled
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce to 2D and 3D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Step 5: 2D Scatter Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2D Projection of Digits Dataset')
plt.colorbar(scatter, label='Digit Label')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=y, cmap='tab10', alpha=0.7)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('PCA - 3D Projection of Digits Dataset')
fig.colorbar(scatter, label='Digit Label')
plt.tight_layout()
plt.show()

# Step 7: Scree Plot (Explained Variance by Each Component)
pca_full = PCA().fit(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Cumulative Explained Variance
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA')
plt.grid(True)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Threshold')
plt.legend()
plt.tight_layout()
plt.show()




# --------------------------------------------

# To understand and apply data preprocessing techniques such as categorical encoding, feature scaling, and binning/discretization using a real-world dataset.


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    header=None,
    names=column_names,
    na_values=' ?'
)

# Display first few rows
df.head()

# Check null values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Label encoding
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])   # '>50K':1 , '<=50K':0
df['sex'] = le.fit_transform(df['sex'])         # 'Male':1 , 'Female':0

# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=[
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'native-country'
], drop_first=True)

# Standard scaling for selected columns
scaler = StandardScaler()
df[['age', 'hours-per-week']] = scaler.fit_transform(df[['age', 'hours-per-week']])

# Min-max scaling for capital gain/loss
minmax = MinMaxScaler()
df[['capital-gain', 'capital-loss']] = minmax.fit_transform(df[['capital-gain', 'capital-loss']])

# Binning age into 4 uniform bins
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
df['age_binned'] = kb.fit_transform(df[['age']])

# Plot income distribution across age bins
sns.countplot(data=df, x='age_binned', hue='income')
plt.title("Income Distribution across Age Bins")
plt.show()



# ---------------------------------


# Exp 9 House Price Prediction using Linear Regression

# =============================================
# Title: House Price Prediction using Linear Regression
# Dataset: Boston Housing Dataset (Kaggle)
# =============================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load and Explore the Dataset
# Make sure the file "BostonHousing.csv" is in the same directory as this notebook
data = pd.read_csv(r"C:\Users\IU\Downloads\BostonHousing.csv")

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing values in dataset:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# Step 3: Visualize the Data

# Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Boston Housing Dataset")
plt.show()

# Scatter plots between key features and target
features_to_plot = ['RM', 'LSTAT', 'PTRATIO', 'DIS']
for feature in features_to_plot:
    plt.figure(figsize=(5,4))
    sns.scatterplot(x=data[feature], y=data['MEDV'])
    plt.title(f'{feature} vs MEDV')
    plt.xlabel(feature)
    plt.ylabel('House Price (MEDV)')
    plt.show()

# Step 4: Feature Selection and Preprocessing
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Model Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

print(f"\nIntercept: {model.intercept_:.2f}")

# Step 8: Visualize Results
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()

# Step 9: Conclusion
print("\nConclusion:")
print("The Linear Regression model was successfully trained to predict house prices.")
print("Model performance evaluated using RMSE and R² score.")
print("Visualization shows a good alignment between predicted and actual prices.")


# ------------------------------

# Exp10 Classification using Logistic Regression and k-Nearest Neighbors (k-NN)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

data = pd.read_csv(r'C:\Users\IU\Downloads\SocialNetworkAds.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

sns.countplot(x='Purchased', data=data)
plt.title("Purchase Distribution")
plt.show()

sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=data, palette='coolwarm')
plt.title("Age vs Estimated Salary (Colored by Purchase)")
plt.show()

X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)

print("=== Logistic Regression Results ===")
print("Confusion Matrix:\n", cm_log)
print(f"Accuracy: {acc_log:.2f}")
print(f"Precision: {prec_log:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log))

ConfusionMatrixDisplay(cm_log, display_labels=['Not Purchased', 'Purchased']).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)

print("\n=== k-NN Results ===")
print("Confusion Matrix:\n", cm_knn)
print(f"Accuracy: {acc_knn:.2f}")
print(f"Precision: {prec_knn:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

ConfusionMatrixDisplay(cm_knn, display_labels=['Not Purchased', 'Purchased']).plot()
plt.title("Confusion Matrix - k-NN")
plt.show()

def plot_decision_boundary(X_set, y_set, model, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('lightcoral', 'lightgreen'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.show()

plot_decision_boundary(X_train, y_train, log_reg, "Decision Boundary - Logistic Regression (Training set)")
plot_decision_boundary(X_test, y_test, log_reg, "Decision Boundary - Logistic Regression (Test set)")
plot_decision_boundary(X_train, y_train, knn, "Decision Boundary - k-NN (Training set)")
plot_decision_boundary(X_test, y_test, knn, "Decision Boundary - k-NN (Test set)")

print("\n=== Model Performance Comparison ===")
print(f"Logistic Regression -> Accuracy: {acc_log:.2f}, Precision: {prec_log:.2f}")
print(f"k-NN               -> Accuracy: {acc_knn:.2f}, Precision: {prec_knn:.2f}")

print("\nConclusion:")
print("Both Logistic Regression and k-NN models were trained and evaluated.")
print("Performance varies slightly depending on data distribution and feature scaling.")
print("Logistic Regression performs well on linearly separable data, while k-NN adapts to complex decision boundaries.")
