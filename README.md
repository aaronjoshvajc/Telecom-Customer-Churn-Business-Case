#  Customer Churn Prediction - No-Churn Telecom (PRCL-0017)

A machine learning project to predict customer churn for No-Churn Telecom — enabling targeted retention campaigns using churn risk scores and explainable AI.


##  Project Overview

This project helps **No-Churn Telecom** (a European telecom provider) reduce its churn rate (over 10%) by predicting which customers are likely to leave using machine learning.

###  Goals
1. Identify the **key factors causing churn**
2. Create a **churn risk score** to prioritize retention efforts
3. Classify customers using a new **`CHURN_FLAG`** variable: `YES` (high churn risk) or `NO` (low risk)


##  Machine Learning Pipeline

| Step                | Details |
|---------------------|---------|
| **Data Source**     | SQL DB (`project_telecom.telecom_churn_data`) — 4,617 records |
| **Preprocessing**   | Dropped `Phone`, `State`; encoded `Churn`, scaled features |
| **Feature Engineering** | Created churn probability + `CHURN_FLAG` |
| **Modeling**        | XGBoost Classifier |
| **Evaluation**      | Confusion matrix, ROC AUC, precision, recall, F1-score |
| **Interpretability**| SHAP for feature importance |
| **Deliverables**    | CSV with churn predictions and a dashboard-ready format |


##  Dataset Snapshot

| Column               | Description                          |
|----------------------|--------------------------------------|
| `Account_Length`     | Days of customer tenure              |
| `VMail_Message`      | Number of voicemail messages         |
| `Day/Eve/Night Mins` | Usage across different periods       |
| `CustServ_Calls`     | Number of customer service calls     |
| `Churn`              | Target variable: `True.` or `False.` |

After cleaning:
- `Churn` converted to binary: `1` = churned, `0` = not churned
- `CHURN_FLAG`: `YES` = likely to churn, `NO` = safe


##  Model Performance (XGBoost)

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 96%       |
| Precision     | 87%       |
| Recall        | 82%       |
| F1-Score      | 84%       |
| ROC AUC       | 0.83      |

>  Balanced and business-relevant — high recall means fewer lost customers.


##  SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

###  Feature Importance (via SHAP)
Top features influencing churn:
- `CustServ_Calls`
- `International_Calls`
- `Day_Mins`
- `International_Charge`

SHAP helps build stakeholder trust by showing **why** the model predicts churn.


##  Output Files

- `churn_predictions_final.csv` — full prediction report:
  - `actual_churn`, `churn_risk_score`, `CHURN_FLAG (YES/NO)`
- `churn_risk_customers.csv` — filtered list of customers likely to churn


##  Tech Stack

- Python 3.12
- Jupyter Notebook
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `shap`
- `StandardScaler`, `train_test_split`


##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook PRCL-0017-Customer Churn Business case.ipynb
