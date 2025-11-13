import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import skew


print("\nStep 1: Describe the size of the sample (how many rows/observations")

print("=" *100)

# Load the dataset
df = pd.read_csv('Dataset/StudentsPerformance.csv')

# Step 1: Describe the size of the sample
num_rows = df.shape[0]   # number of rows (observations)
num_columns = df.shape[1]  # number of columns (features)

print(f"Number of observations (rows): {num_rows}")
print(f"Number of variables (columns): {num_columns}\n")

# Step 2: Describe the centre of the data (mean, median of scores)
print("\nStep 2: Describe the centre of the data (mean, median of scores).")
print("=" *100)

mean_math = df['math score'].mean()
median_math = df['math score'].median()

print("math score")
print(f"Mean: {round(mean_math, 2)}")
print(f"Median: {round(median_math, 2)}\n")

mean_reading = df['reading score'].mean()
median_reading = df['reading score'].median()

print("reading score")
print(f"Mean: {round(mean_reading, 2)}")
print(f"Median: {round(median_reading, 2)}\n")

mean_writing = df['writing score'].mean()
median_writing = df['writing score'].median()

print("writing score")
print(f"Mean: {round(mean_writing, 2)}")
print(f"Median: {round(median_writing, 2)}\n")

# Step 3: Describe the spread of the data (range, variance, standard deviation of scores)
print("\nStep 3: Describe the spread of the data (variance, standard deviation of scores).")
print("=" *100)

var_math = df['math score'].var()
std_math = df['math score'].std()   
print("math score")
print(f"Variance: {round(var_math, 2)}")
print(f"Standard Deviation: {round(std_math, 2)}\n")

var_reading = df['reading score'].var()
std_reading = df['reading score'].std()
print("reading score")
print(f"Variance: {round(var_reading, 2)}") 
print(f"Standard Deviation: {round(std_reading, 2)}\n")

var_writing = df['writing score'].var()
std_writing = df['writing score'].std()
print("writing score")
print(f"Variance: {round(var_writing, 2)}")
print(f"Standard Deviation: {round(std_writing, 2)}")

# Step 4: Assess the shape of the data distribution (use histograms and skewness scores)
print("\nStep 4: Assess the shape of the data distribution (use histograms and skewness scores).")
print("=" *100)

# Plot histograms for numerical columns
df.hist(bins=10, figsize=(12, 8))
plt.suptitle("Distribution of Numerical Variables", fontsize=16)
plt.show()

# Calculate skewness for numerical columns

'''Skewness measures asymmetry:

0 → perfectly symmetrical (normal)

> 0 → right-skewed (tail to the right)

< 0 → left-skewed (tail to the left)'''

skewness = df.skew(numeric_only=True)

print("Skewness of numerical columns:")

for col, skew_value in skewness.items():
    skew_value = round(skew_value, 2)
    if skew_value > 0:
        print(f"Math Score: {skew_value}, indicating the distribution is right-skewed.")
    elif skew_value < 0:
        print(f"Reading Score: {skew_value}, indicating the distribution is left-skewed.")
    else:
        print(f"Writing Score: {skew_value}, indicating the distribution is symmetrical.")




# Step 5: Compare data from different groups (e.g., male vs female performance).
print("\nStep 5: Compare data from different groups (e.g., male vs female performance).")
print("=" *100)


import seaborn as sns

 
# Visualization: Boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='gender', y='math score', data=df, palette='Set2', legend=False, showfliers=False, hue='gender')
plt.title('Comparison of Math Score Performance by Gender')
plt.xlabel('Gender')
plt.ylabel('Performance Score')
plt.show()

# Numerical Summary: Mean values by group
numeric_cols = ['math score', 'reading score', 'writing score']  # change these to your column names
group_means = df.groupby('gender')[numeric_cols].mean()

print("Mean values by group:")
print(group_means)

# --- Plot comparison ---
group_means.plot(kind='bar', figsize=(8,6))
plt.title('Mean Comparison Across Gender Groups')
plt.xlabel('Gender')
plt.ylabel('Mean Value')
plt.legend(title='Numeric Fields')
plt.xticks(rotation=0)
plt.show()