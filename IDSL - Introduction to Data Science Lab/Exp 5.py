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
