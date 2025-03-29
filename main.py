import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import streamlit as st
import time

# Extract the SQLite file from the zip
zip_path = '/Users/ngoubimaximilliandiamgha/Desktop/database.sqlite.zip'
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/Users/ngoubimaximilliandiamgha/Desktop/')

# Path to your extracted SQLite file
sqlite_db_path = '/Users/ngoubimaximilliandiamgha/Desktop/database.sqlite'

# Connect to SQLite
conn = sqlite3.connect(sqlite_db_path)

# PostgreSQL connection details
postgres_user = 'postgres'
postgres_password = 'hope'
postgres_host = 'localhost'
postgres_port = '5432'
postgres_db = 'DataAnalysisusingSQL'

postgres_conn_str = f'postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}'
postgres_engine = create_engine(postgres_conn_str)

# Transfer tables to PostgreSQL with loading indicator
start_time = time.time()
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
st.spinner("Transferring tables to PostgreSQL...")
for table in tables:
    df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
    df.to_sql(table, postgres_engine, if_exists='replace', index=False)
    print(f'Table {table} transferred successfully.')
st.success(f"Data transfer completed in {time.time() - start_time:.2f} seconds")

# Load Match data
matches_df = pd.read_sql("SELECT * FROM Match;", conn)

# Data preprocessing
matches_df.dropna(inplace=True)
matches_df['goal_difference'] = matches_df['home_team_goal'] - matches_df['away_team_goal']
matches_df['result'] = np.where(matches_df['goal_difference'] > 0, 1, 0)

# Regression Analysis
X_reg = matches_df[['home_team_goal', 'away_team_goal']]
y_reg = matches_df['goal_difference']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

# Classification Models
X_clf = matches_df[['home_team_goal', 'away_team_goal']]
y_clf = matches_df['result']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Support Vector Machine": SVC()
}

accuracies = {}
precisions = {}
recalls = {}
f1s = {}
matrices = {}
for name, model in models.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)
    accuracies[name] = accuracy_score(y_test_clf, y_pred)
    precisions[name] = precision_score(y_test_clf, y_pred)
    recalls[name] = recall_score(y_test_clf, y_pred)
    f1s[name] = f1_score(y_test_clf, y_pred)
    matrices[name] = confusion_matrix(y_test_clf, y_pred)

# Streamlit App
st.title("Football Match Analysis Dashboard")

# Regression Section
st.header("Regression: Predict Goal Difference")
home_goal = st.number_input("Enter Home Team Goals", min_value=0, step=1, key='reg_home')
away_goal = st.number_input("Enter Away Team Goals", min_value=0, step=1, key='reg_away')
if st.button("Predict Goal Difference"):
    predicted_diff = reg_model.predict([[home_goal, away_goal]])[0]
    st.success(f"Predicted Goal Difference: {predicted_diff:.2f}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test_reg, y=reg_model.predict(X_test_reg), ax=ax)
    ax.set_xlabel("Actual Goal Difference")
    ax.set_ylabel("Predicted Goal Difference")
    ax.set_title("Regression Prediction vs Actual")
    st.pyplot(fig)

# Classification Section
st.header("Classification: Predict Match Outcome")
classifier = st.selectbox("Choose Model", list(models.keys()))
home_goal_cls = st.number_input("Enter Home Team Goals", min_value=0, step=1, key='cls_home')
away_goal_cls = st.number_input("Enter Away Team Goals", min_value=0, step=1, key='cls_away')
if st.button("Predict Match Outcome"):
    outcome = models[classifier].predict([[home_goal_cls, away_goal_cls]])[0]
    result = "Home Win" if outcome == 1 else "Away Win or Draw"
    st.success(f"Predicted Outcome: {result} (Model: {classifier})")
    fig, ax = plt.subplots()
    sns.heatmap(matrices[classifier], annot=True, fmt='d', cmap='viridis', ax=ax)
    ax.set_title(f"{classifier} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Model Accuracy Overview
st.header("Model Performance Metrics")
for name in models:
    st.subheader(name)
    st.write(f"Accuracy: {accuracies[name]:.2%}")
    st.write(f"Precision: {precisions[name]:.2%}")
    st.write(f"Recall: {recalls[name]:.2%}")
    st.write(f"F1 Score: {f1s[name]:.2%}")

# Close connections
conn.close()
postgres_engine.dispose()

print("All tasks completed successfully!")
