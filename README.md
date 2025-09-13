# Bank Customer Churn Prediction Dashboard

This project predicts bank customer churn (whether a customer will leave the bank) using Machine Learning (Decision Trees, XGBoost) and provides an interactive Streamlit dashboard for visualizing insights and making predictions.  

---

## Dataset

We are using the Bank Customer Churn Prediction Dataset from Kaggle:  
`/kaggle/input/bank-customer-churn-prediction-dataset/Churn_Modelling.csv`

This dataset contains:
- RowNumber, CustomerId, Surname (Identifiers)
- CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Exited (Target: 1 = Churn, 0 = Stay)

---

## Libraries Used & Their Purpose

| Library / Module | Purpose |
|------------------|--------|
| pandas | Load, clean, and manipulate dataset |
| numpy | Math operations and array manipulation |
| seaborn | Statistical visualizations (heatmap for feature correlation) |
| matplotlib.pyplot | Static plots (histograms, bar charts) |
| plotly.express | Interactive charts (zoom, hover, filter) |
| streamlit | Build interactive dashboard for churn analysis |
| sklearn.tree.DecisionTreeClassifier | Train ML model for churn prediction |
| sklearn.model_selection.train_test_split | Split data into training/testing sets |
| sklearn.preprocessing.MinMaxScaler | Normalize features for ML model |
| pickle | Save and load trained model |
| xgboost (install separately) | More accurate boosting algorithm |
| graphviz (install separately) | Visualize decision trees |

---

## Installation

Install the dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn plotly streamlit xgboost graphviz
