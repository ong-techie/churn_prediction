import streamlit as st
import pandas as pd
import joblib
import os

# Load trained pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "notebook", "model_pipeline.sav")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📉 Telco Customer Churn Prediction")

st.write("Enter customer details to predict churn probability.")

# -------- Input UI --------
input_data = {
    'gender': st.selectbox('Gender', ['Male', 'Female']),
    'SeniorCitizen': st.selectbox('Senior Citizen', [0, 1]),
    'Partner': st.selectbox('Partner', ['Yes', 'No']),
    'Dependents': st.selectbox('Dependents', ['Yes', 'No']),
    'tenure': st.slider('Tenure (months)', 0, 72, 1),
    'PhoneService': st.selectbox('Phone Service', ['Yes', 'No']),
    'MultipleLines': st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service']),
    'InternetService': st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': st.selectbox('Online Security', ['Yes', 'No', 'No internet service']),
    'OnlineBackup': st.selectbox('Online Backup', ['Yes', 'No', 'No internet service']),
    'DeviceProtection': st.selectbox('Device Protection', ['Yes', 'No', 'No internet service']),
    'TechSupport': st.selectbox('Tech Support', ['Yes', 'No', 'No internet service']),
    'StreamingTV': st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service']),
    'StreamingMovies': st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service']),
    'Contract': st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': st.selectbox('Paperless Billing', ['Yes', 'No']),
    'PaymentMethod': st.selectbox(
        'Payment Method',
        [
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ]
    ),
    'MonthlyCharges': st.number_input('Monthly Charges', 0.0, 200.0, 70.0),
    'TotalCharges': st.number_input('Total Charges', 0.0, 10000.0, 1000.0)
}

df_input = pd.DataFrame([input_data])

# ---- Feature engineering (MUST match training) ----

# tenure_bucket
df_input['tenure_bucket'] = pd.cut(
    df_input['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
)

# charges_per_month
df_input['charges_per_month'] = (
    df_input['TotalCharges'] / (df_input['tenure'] + 1)
)



# -------- Prediction --------
if st.button("Predict Churn"):
    proba = model.predict_proba(df_input)[0][1]
    prediction = "YES – High Risk" if proba >= 0.5 else "NO – Low Risk"

    st.subheader("Prediction Result")
    st.success(f"**{prediction}**")
    st.write(f"Churn Probability: **{proba:.2%}**")

    st.progress(min(int(proba * 100), 100))
