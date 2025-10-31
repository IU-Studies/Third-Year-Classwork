# Basic Lib Experiment

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





# Implement A* Algorithm

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




# Basic Calculations


# Basic Arithmetic and Statistical Calculations
import numpy as np

print(" ")

# Arithmetic Operations
a = 4664
b = 29

print("We are taking a as",a,"and b as",b)

print("Arithmetic Operations:")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Modulus: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")
print()

# Statistical Calculations
data = [546, 23210, 12561340, 4150325, 210]
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

print("Statistical Calculations:")
print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")



