# ⚽ FOOTBALL MATCH AI INSIGHTS

An interactive Streamlit dashboard for analyzing football match data using regression and classification machine learning models. This project combines SQLite-to-PostgreSQL migration, model training, and beautiful visualization — all in one app.

---

## 🚀 FEATURES

- 📊 **Regression**: Predict goal differences between home and away teams
- 🤖 **Classification**: Predict match outcome using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Machine (SVM)
- 📈 **Model Visualizations**:
  - Scatter plot for regression accuracy
  - Confusion matrices for classification
- 📋 **Performance Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- 🗃️ **Data Migration**:
  - Extracts from SQLite ZIP
  - Transfers to PostgreSQL automatically
- 🧠 **Streamlit Interface**:
  - Predict outcomes from user input
  - View model comparison and visual results

---

## 📂 PROJECT STRUCTURE

```bash
Football-match-ai-insights/
├── main.py               # Streamlit app with data pipeline & visualizations
├── database.sqlite.zip   # Original dataset
├── match_with_teams.csv  # Sample exported table (optional)
├── README.md             # Project documentation (this file)
└── requirements.txt      # Python dependencies (optional)
