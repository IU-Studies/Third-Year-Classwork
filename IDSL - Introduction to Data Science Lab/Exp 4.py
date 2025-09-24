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
