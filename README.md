# Market-Predictive-Ai

# Import libraries
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from ipywidgets import interact, widgets
import plotly.express as px
%matplotlib inline

# Load dataset and handle mixed types
df = pd.read_csv(r"D:\Semester 5\Train_Dataset.csv", low_memory=False)

# Fill missing values: numeric -> 0, categorical -> mode
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) if df[col].dtype in ['int64','float64'] else df[col].fillna(df[col].mode()[0])

# EDA: Client Income and Age distribution
if 'Client_Income' in df.columns: plt.hist(pd.to_numeric(df['Client_Income'], errors='coerce').dropna(), bins=30, color='skyblue', edgecolor='black'); plt.title("Client Income"); plt.show()
if 'Age_Days' in df.columns: plt.hist(pd.to_numeric(df['Age_Days'], errors='coerce')/-365, bins=30, color='orange', edgecolor='black'); plt.title("Age Distribution"); plt.show()

# Active Loan counts
if 'Active_Loan' in df.columns: sns.countplot(x='Active_Loan', data=df); plt.title("Active Loan Counts"); plt.show()

# Average Credit for car owners
car_col = next((c for c in df.columns if 'car' in c.lower()), None); credit_col = next((c for c in df.columns if 'credit' in c.lower()), None)
if car_col and credit_col: print("Avg Credit (Car Owners):", df[df[car_col]=='Y'][credit_col].mean())

# Correlation between Credit and Loan Annuity
loan_col = next((c for c in df.columns if 'annuity' in c.lower()), None)
if credit_col and loan_col: print("Correlation Credit vs Loan:", pd.to_numeric(df[credit_col].astype(str).str.replace(r'[^0-9.]','', regex=True), errors='coerce').corr(pd.to_numeric(df[loan_col].astype(str).str.replace(r'[^0-9.]','', regex=True), errors='coerce')))

# Most common education
if 'Education' in df.columns: print("Most common education:", df['Education'].mode()[0])

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(exclude='number').columns: df[col] = le.fit_transform(df[col])

# Features & Target, scale and split
X = pd.DataFrame(StandardScaler().fit_transform(df.drop("Default", axis=1)), columns=df.drop("Default", axis=1).columns)
y = df["Default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train models interactively
def train_models(max_depth=5, n_estimators=100):
    log = LogisticRegression(max_iter=1000); log.fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42); dt.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42); rf.fit(X_train, y_train)
    print("Logistic ROC-AUC:", roc_auc_score(y_test, log.predict_proba(X_test)[:,1]), " | Decision Tree F1:", f1_score(y_test, dt.predict(X_test)), " | Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
    plt.figure(figsize=(10,5)); plot_tree(dt, feature_names=X.columns, class_names=["No Default","Default"], filled=True); plt.show()
    sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt="d", cmap="Blues"); plt.show()
    top = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10); px.bar(top, title="Top 10 Features").show()

# Interactive sliders for Decision Tree depth & Random Forest estimators
interact(train_models, max_depth=widgets.IntSlider(value=5,min=1,max=20,step=1,description="Tree Depth"), n_estimators=widgets.IntSlider(value=100,min=10,max=500,step=10,description="Forest Size"))
