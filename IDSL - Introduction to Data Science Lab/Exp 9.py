# 1 Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2 Load and Explore the Dataset
df = pd.read_csv(r"C:\Users\IU\Downloads\Housing.csv")

# Display first few rows
print("First 5 Rows:")
print(df.head())

# Display basic info and statistics
print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# 3 Data Preprocessing

# Convert categorical columns to numeric (yes/no → 1/0, furnishingstatus → one-hot encoding)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
               'airconditioning', 'prefarea']

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

print("\nData after encoding:")
print(df.head())

# 4 Feature Selection
X = df.drop('price', axis=1)
y = df['price']

# 5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6 Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Model summary
print("\nIntercept (β₀):", model.intercept_)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients:")
print(coeff_df)

# 7 Evaluate the Model
y_pred = model.predict(X_test)

# RMSE and R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 8 Visualize Results
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()

# 9 Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Housing Dataset")
plt.show()
