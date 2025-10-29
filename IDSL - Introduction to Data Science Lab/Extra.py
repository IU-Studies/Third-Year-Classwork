# Exp 5

import pandas as pd  # for data manipulation and analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
tips = pd.read_csv(r"C:\Users\IU\Downloads\tips.csv")

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




#Exp 7 
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



#Exp 8 
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
