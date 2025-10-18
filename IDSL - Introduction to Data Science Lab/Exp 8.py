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
