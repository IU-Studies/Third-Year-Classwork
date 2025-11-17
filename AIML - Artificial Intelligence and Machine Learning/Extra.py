# BFS Exp1
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': ['2', '13', '14'],
    '9': ['10'],
    '13': ['10'],
    '14': [],
    '10': ['11'],
    '11': []
}


def bfs(graph, start):
    visited = set()
    queue = [start]
    print("BFS Sequence is as follows:", end=" ")

    while queue:
        current = queue.pop(0)
        if current not in visited:
            print(current, end=" ")
            visited.add(current)
            queue.extend(graph[current])


bfs(graph, '5')




#-------------------------------------------------------





# A* Algo Exp2
import heapq

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(node, grid):
    x, y = node
    rows, cols = len(grid), len(grid[0])
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-neighbors
        nx, ny = x+dx, y+dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            yield (nx, ny)

def reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))

def astar(grid, start, goal):
    if start == goal:
        return [start]
    open_heap = []
    g = {start: 0}
    f = {start: manhattan(start, goal)}
    came_from = {}

    heapq.heappush(open_heap, (f[start], start))

    closed = set()
    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return reconstruct(came_from, current)
        closed.add(current)

        for n in neighbors(current, grid):
            tentative_g = g[current] + 1  # cost of moving to neighbor = 1
            if tentative_g < g.get(n, float('inf')):
                came_from[n] = current
                g[n] = tentative_g
                f[n] = tentative_g + manhattan(n, goal)
                heapq.heappush(open_heap, (f[n], n))
    return None  # no path

# Example usage
if __name__ == "__main__":
    grid = [
        [0,0,0,0],
        [0,1,1,0],
        [0,0,0,0],
        [0,1,0,0],
    ]
    start = (0,0)
    goal = (3,3)
    path = astar(grid, start, goal)
    print("Path:", path)





#-------------------------------------------------------





# Tic Tac Toe Exp3
# Function to draw the Tic-Tac-Toe board
def draw_board(cells):
    def row(i):
        print(f"{cells[i]} | {cells[i + 1]} | {cells[i + 2]}")
    sep = "--+---+--"
    row(0)
    print(sep)
    row(3)
    print(sep)
    row(6)

# Initialize the board with cell numbers 0–8
cells = [str(i) for i in range(9)]

# Define all possible winning combinations
win = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
       {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
       {0, 4, 8}, {2, 4, 6}]

# Lists to store moves of Player 1 and Player 2
list1 = []
list2 = []
won = False

# Game loop for maximum 9 turns
for turn in range(9):
    draw_board(cells)

    # Decide current player
    if turn % 2 == 0:
        currp = 1
    else:
        currp = 2

    # Assign move list and symbol for current player
    if currp == 1:
        plist = list1
        symbol = 'X'
    else:
        plist = list2
        symbol = 'O'

    # Take player input for move
    move = int(input(f"Player {currp} turn ({symbol}): "))

    # Validate move
    if move < 0 or move > 8 or cells[move] in ['X', 'O']:
        print("Invalid move, please try again")
        continue

    # Update board and store move
    cells[move] = symbol
    plist.append(move)

    # Check for winning condition
    if len(plist) >= 3:
        for i in win:
            if i.issubset(set(plist)):
                draw_board(cells)
                print(f"Player {currp} won!")
                won = True
                exit()

# If no winner, declare draw
draw_board(cells)
if not won:
    print("It's a draw")




#-------------------------------------------------------




    

#Basic Python Libraries Exp4

import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# ----- Math Functions -----
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

# ----- Load Dataset -----
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("\nFirst 5 Records:\n", df.head())
print("\nNull Values in Dataset:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# ----- NumPy Statistics -----
sepal_length = df['sepal_length'].values

mean_val = np.mean(sepal_length)
median_val = np.median(sepal_length)
std_dev = np.std(sepal_length)

print("Mean (NumPy):", round(mean_val, 2))
print("Median (NumPy):", round(median_val, 2))
print("Standard Deviation (NumPy):", round(std_dev, 2))

# ----- SciPy Statistics -----
mode_val = stats.mode(sepal_length, keepdims=True)
corr_val = stats.pearsonr(df['sepal_length'], df['petal_length'])

print("Mode (SciPy):", mode_val.mode[0], "(count =", mode_val.count[0], ")")
print("Correlation between Sepal Length & Petal Length (SciPy):", round(corr_val.correlation, 2))
print()

# ----- Line Plot -----
plt.figure(figsize=(6, 4))
plt.plot(df['sepal_length'], color='blue')
plt.title("Line Plot of Sepal Length")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length")
plt.grid(True)
plt.show()

# ----- Bar Plot -----
mean_by_species = df.groupby('species')['sepal_length'].mean()

plt.figure(figsize=(6, 4))
mean_by_species.plot(kind='bar', color=['orange', 'green', 'blue'])
plt.title("Average Sepal Length by Species")
plt.ylabel("Mean Sepal Length")
plt.show()

# ----- Scatter Plot -----
plt.figure(figsize=(6, 4))
plt.scatter(df['sepal_length'], df['petal_length'], color='red')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# ----- Histogram -----
plt.figure(figsize=(6, 4))
plt.hist(df['sepal_length'], bins=10, color='purple', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()






#------------------------------------------------------


#Basic Arithmetic Exp5
# Basic Arithmetic and Statistical Calculations
import numpy as np

# Arithmetic Operations
a = 15
b = 4

print("Arithmetic Operations:")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Modulus: {a} % {b} = {a % b}")


print(f"Exponentiation: {a} ** {b} = {a ** b}")
print()

# Statistical Calculations
data = [10, 20, 30, 40, 50]

mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

print("Statistical Calculations:")
print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")






#-------------------------------------------------------





# Basic data preprocessing Exp6
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load Dataset
url = r"C:\Users\mayur\Downloads\titanic.csv"
df = pd.read_csv(url)

print("First 5 rows of dataset:\n", df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

# ----- Handle Missing Values -----
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)   # must reassign!

print("\nMissing Values After Handling:\n", df.isnull().sum())

# ----- Label Encoding -----
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# ----- One-hot Encoding -----
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nData after Encoding:\n", df.head())

# ----- Normalization -----
numeric_cols = ['Age', 'Fare']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nData after Normalization:\n", df[numeric_cols].head())

print("\nFinal Processed Dataset:\n", df.head())

# Save processed file
df.to_csv("titanic_preprocessed.csv", index=False)




#-------------------------------------------------------





#ROC Curve Exp7

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("AUC:", roc_auc)

# ROC Curve Plot
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()





#-------------------------------------------------------




# Bayesian Classifier Exp8

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%\n")

print("Confusion Matrix:")
print(conf_matrix)

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix, annot=True, cmap="Blues", fmt="d",
    xticklabels=iris.target_names, yticklabels=iris.target_names
)
plt.title("Confusion Matrix - Naive Bayes Classifier")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))




#---------------------------------------------------------------------


# Kmeans and Silhouette Exp9

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
X_scaled = StandardScaler().fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title("K-Means Clustering (Wine Dataset)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()




#-----------------------------------------------------------

#PCA Exp10

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_.sum())

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

score = silhouette_score(X_pca, labels)
print("Silhouette Score:", score)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title("K-Means Clustering on PCA-Reduced Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("Actual Wine Classes (After PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
