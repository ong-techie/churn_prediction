# Customer Churn Prediction 🧠📉

Customer churn prediction is critical for subscription-based businesses to proactively identify customers who are likely to leave and take timely retention actions.
This project uses machine learning (XGBoost + Scikit-learn pipelines) to predict churn probability from customer behavior and service usage data, and exposes the model via an interactive Streamlit application.

## 🔍 Problem Statement

Churn rate reflects the percentage of customers who stop using a company’s service. This project focuses on:

- Identifying customers likely to churn
- Improving retention strategies
- Enhancing product/service offerings based on customer behavior

## 🎯 Project Goals

- Predict the likelihood of customer churn using historical data
- Use Pipeline + GridSearchCV for consistent preprocessing and modeling
- Support real-time (online) predictions

## 🧰 Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Streamlit** (for interactive UI)
- **Joblib** (for model serialization)

## 🧱 Workflow

1. **Problem Definition**: Understand churn causes and business goals.
2. **Data Acquisition**: Collect data from CRM systems, analytics tools, and feedback.
3. **Data Preprocessing**: Clean, transform, and explore the dataset for modeling.
4. **Modeling**: Train and evaluate models (e.g., Logistic Regression, Random Forest, etc.).
5. **Deployment**: Provide real-time prediction functionality via Streamlit app.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ong-techie/churn_predictor.git
   cd churn_predictor

2. Install dependencies:
    pip install -r requirements.txt

3. Launch the Streamlit app:
    streamlit run p.py

📦 Features
* Load customer data and explore it visually
* Make predictions for individual customers or in batch
* View churn probability and risk factors
* Clean and intuitive UI built with Streamlit

📊 Prediction Modes
* Online Prediction: Enter a single customer's data manually for instant prediction.

🏆 Model Performance
| Model               | ROC-AUC        |
| ------------------- | -------------- |
| Logistic Regression | ~0.83          |
| Decision Tree       | ~0.83          |
| Random Forest       | ~0.85          |
| **XGBoost (Final)** | **~0.86–0.88** |

📁 Project Structure
churn_predictor/
│
├── data/
│   └── churn.csv
│
├── notebook/
│   └── model_pipeline.sav
│   └── model_train.ipynb
│ 
├── app.py
├── requirements.txt
└── README.md
