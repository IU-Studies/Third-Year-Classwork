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
