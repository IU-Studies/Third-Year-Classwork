import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
num = 5.6
x = 5
print("ceil(", num, ") =", math.ceil(num))
print("floor(", num, ") =", math.floor(num))
print("fabs(-", num, ") =", math.fabs(-num))
print("factorial(", x, ") =", math.factorial(x))
print("sqrt(", x, ") =", math.sqrt(x))
print("copysign(5, -3) =", math.copysign(5, -3))
print("log(10) =", math.log(10))
print()
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print("\nFirst 5 Records:\n", df.head())
print("\nNull Values in Dataset:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())
sepal_length = df['sepal_length'].values
mean_val = np.mean(sepal_length)
median_val = np.median(sepal_length)
std_dev = np.std(sepal_length)
print("Mean (NumPy):", round(mean_val, 2))
print("Median (NumPy):", round(median_val, 2))
print("Standard Deviation (NumPy):", round(std_dev, 2))
mode_val = stats.mode(sepal_length, keepdims=True)
corr_val = stats.pearsonr(df['sepal_length'], df['petal_length'])
print("Mode (SciPy):", mode_val.mode[0], "(count =", mode_val.count[0], ")")
print("Correlation between Sepal Length & Petal Length (SciPy):", round(corr_val.correlation, 2))
print()

plt.figure(figsize=(6, 4))
plt.plot(df['sepal_length'], color='blue')
plt.title("Line Plot of Sepal Length")

plt.xlabel("Sample Index")
plt.ylabel("Sepal Length")
plt.grid(True)
plt.show()
mean_by_species = df.groupby('species')['sepal_length'].mean()
plt.figure(figsize=(6, 4))
mean_by_species.plot(kind='bar', color=['orange', 'green', 'blue'])
plt.title("Average Sepal Length by Species")
plt.ylabel("Mean Sepal Length")
plt.show()
plt.figure(figsize=(6, 4))
plt.scatter(df['sepal_length'], df['petal_length'], color='red')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
plt.figure(figsize=(6, 4))
plt.hist(df['sepal_length'], bins=10, color='purple', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()
