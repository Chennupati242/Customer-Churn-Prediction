import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Creating Synthetic Customer Data
np.random.seed(42)
n_customers = 1000

data = {
    'Monthly_Spend': np.random.normal(2000, 800, n_customers),
    'Payment_Delay_Days': np.random.randint(0, 30, n_customers),
    'Customer_Service_Calls': np.random.randint(0, 10, n_customers),
    'Years': np.random.uniform(1, 15, n_customers),
    'Churn': np.random.choice([0, 1], size=n_customers, p=[0.8, 0.2]) 
}

# Adding logic
df = pd.DataFrame(data)
df.loc[df['Customer_Service_Calls'] > 5, 'Churn'] = 1 

#Train the Machine Learning Model
X = df.drop('Churn', axis=1) # Features (The clues)
y = df['Churn']              # Target (The answer: stayed or left)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Check Accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 4. Visualize "Why" people leave (Feature Importance)
importances = pd.Series(model.feature_importances_, index=X.columns)

# Create the plot
plt.figure(figsize=(10, 6))
importances.nlargest(4).plot(kind='barh', color='#0072CE')

plt.title('Why are Customers Churning?')
plt.xlabel('Importance Score')

plt.tight_layout() 

plt.savefig('churn_factors.png')
plt.show()
print("Chart saved as churn_factors.png with fixed labels!")