
# ===============================
# LEVEL 2 - ALL TASKS
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset .csv")
df['Cuisines'] = df['Cuisines'].fillna("Not Specified")

# Task 1: Table Booking & Online Delivery
print("Table Booking %:\n", df['Has Table booking'].value_counts(normalize=True)*100)
print("Online Delivery %:\n", df['Has Online delivery'].value_counts(normalize=True)*100)
print("Average Rating by Booking:\n", df.groupby('Has Table booking')['Aggregate rating'].mean())

# Task 2: Price Range Analysis
print("Most Common Price Range:\n", df['Price range'].value_counts())
avg_price = df.groupby('Price range')['Aggregate rating'].mean()
print("Average Rating per Price Range:\n", avg_price)

plt.figure()
avg_price.plot(kind='bar')
plt.title("Average Rating by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Average Rating")
plt.show()

# Task 3: Feature Engineering
df['Name_Length'] = df['Restaurant Name'].apply(len)
df['Has_Table_Booking'] = df['Has Table booking'].map({'Yes':1, 'No':0})
df['Has_Online_Delivery'] = df['Has Online delivery'].map({'Yes':1, 'No':0})

print(df[['Name_Length','Has_Table_Booking','Has_Online_Delivery']].head())
