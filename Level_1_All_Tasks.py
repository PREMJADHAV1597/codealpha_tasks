
# ===============================
# LEVEL 1 - ALL TASKS
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset .csv")

# Task 1: Data Exploration
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
df['Cuisines'] = df['Cuisines'].fillna("Not Specified")

plt.figure()
df['Aggregate rating'].hist(bins=20)
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Task 2: Descriptive Analysis
print("\nStatistical Summary:\n", df.describe())
print("\nTop Cities:\n", df['City'].value_counts().head(10))
print("\nTop Cuisines:\n", df['Cuisines'].value_counts().head(10))

# Task 3: Geospatial Analysis
plt.figure()
plt.scatter(df['Longitude'], df['Latitude'])
plt.title("Restaurant Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
