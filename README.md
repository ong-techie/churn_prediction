# Customer Churn Prediction 🧠📉

Customer churn prediction is critical for subscription-based businesses to identify and retain unhappy customers. This machine learning-based project helps forecast customer attrition by analyzing behavioral and historical data to generate actionable insights.

## 🔍 Problem Statement

Churn rate reflects the percentage of customers who stop using a company’s service. This project focuses on:

- Identifying customers likely to churn
- Improving retention strategies
- Enhancing product/service offerings based on customer behavior

## 🎯 Project Goals

- Predict the likelihood of customer churn using historical data
- Enable both **online** (real-time) and **batch** predictions
- Provide a user-friendly interface for data input and visualization

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
5. **Deployment**: Provide real-time and batch prediction functionality via Streamlit app.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/churn_predictor.git
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

* Batch Prediction: Upload a CSV file to get churn predictions for multiple customers at once.


![App Demo](streamlit-app.gif)
