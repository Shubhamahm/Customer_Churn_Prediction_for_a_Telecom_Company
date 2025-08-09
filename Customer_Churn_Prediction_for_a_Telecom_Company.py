# Telecom companies (like Jio, Airtel, Vodafone, etc.) use churn models to predict which
# customers are likely to leave and take action to retain them. This is widely used across
# other industries too—banking, SaaS, retail, etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data="C://Users//shubh//OneDrive//Desktop//All projects//python codes//Telco_Customer_Churn.csv"

df=pd.read_csv(data)
# print(df)
# print(df.shape)
# print(df.columns)

# Check for missing values
# print(df.isnull().sum())

# Check data types and summary
print(df.info())

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

df.dropna(subset=['TotalCharges'],inplace=True)

print(df['TotalCharges'].dtype)
print(df.isnull().sum())
print(df.shape)

#------- EDA (Exploratory Data Analysis)------------

print(df['Churn'].value_counts)

sns.countplot(data=df,x='Churn')
plt.title('Churn Distribution')
# plt.show()

# tenure vs Churn

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure',hue='Churn',bins=30,kde=True,multiple='stack')
plt.title('Tenure vs Churn')
plt.xlabel('Tenure (months)')
plt.ylabel('Number of Customers')
# plt.show()

# Insight: Most customers who churn have short tenures. 
# This suggests that many users leave early, possibly due to poor onboarding, bad service, 
# or better offers from competitors.

# Contract vs Churn

plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Contract',hue='Churn')
plt.title('Contract Type vs Churn')
plt.xticks(rotation=15)
# plt.show()

# Payment Vs Churn

plt.figure(figsize=(8,5))
sns.countplot(data=df,x='PaymentMethod', hue='Churn')
plt.title('Payment Method vs Churn')
plt.xticks(rotation=15)
# plt.show()

# --------Feature Engineering-----------------#

def tenure_group(tenure):
    if tenure <=12:
        return '0-1 year'
    elif tenure<=24:
        return '1-2 years'
    elif tenure <= 36:
        return '2-3 years'
    elif tenure <= 48:
        return '3-4 years'
    elif tenure <= 60:
        return '4-5 years'
    else:
        return '5+ years'
    
df['tenure_group']=df['tenure'].apply(tenure_group)

print(df['tenure_group'].value_counts())

# Scaling Numerical Values

from sklearn.preprocessing import MinMaxScaler

num_cols=['tenure','MonthlyCharges','TotalCharges']

df['TotalCharges']= pd.to_numeric(df['TotalCharges'],errors='coerce')

df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

scalar=MinMaxScaler()
df[num_cols]=scalar.fit_transform(df[num_cols])

print(df[num_cols].head())

# --------Model Building-------------#

# Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=['customerID'])
X_test = X_test.drop(columns=['customerID'])

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# Build and train logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))