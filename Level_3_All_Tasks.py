
# ===============================
# LEVEL 3 - ALL TASKS
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Dataset .csv")
df['Cuisines'] = df['Cuisines'].fillna("Not Specified")

# Feature Engineering
df['Name_Length'] = df['Restaurant Name'].apply(len)
df['Has_Table_Booking'] = df['Has Table booking'].map({'Yes':1, 'No':0})
df['Has_Online_Delivery'] = df['Has Online delivery'].map({'Yes':1, 'No':0})

# Predictive Modeling
features = ['Average Cost for two', 'Price range', 'Votes',
            'Has_Table_Booking', 'Has_Online_Delivery', 'Name_Length']

X = df[features]
y = df['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name} Results:")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))

# Customer Preference
print("\nTop Cuisines by Votes:\n", df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(10))

plt.figure()
df['Aggregate rating'].hist(bins=20)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()
