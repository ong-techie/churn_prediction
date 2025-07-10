# --- app.py (Updated) ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load model and preprocessing tools
model = joblib.load("notebook/model.sav")

# Import preprocessing function
from preprocessing import preprocess

def main():
    st.title("Telco Customer Churn Prediction")

    image = Image.open("App.jpg")
    mode = st.sidebar.selectbox("Choose prediction mode:", ["Online", "Batch"])
    st.sidebar.image(image, caption="Churn Predictor")

    if mode == "Online":
        st.subheader("Enter customer details")

        inputs = {
            'SeniorCitizen': st.selectbox('Senior Citizen:', ('Yes', 'No')),
            'Dependents': st.selectbox('Dependent:', ('Yes', 'No')),
            'tenure': st.slider('Tenure (months)', 0, 72, 1),
            'PhoneService': st.selectbox('Phone Service:', ('Yes', 'No')),
            'MultipleLines': st.selectbox('Multiple Lines:', ('Yes','No','No phone service')),
            'InternetService': st.selectbox('Internet Service:', ('DSL', 'Fiber optic', 'No')),
            'OnlineSecurity': st.selectbox('Online Security:', ('Yes','No','No internet service')),
            'OnlineBackup': st.selectbox('Online Backup:', ('Yes','No','No internet service')),
            'TechSupport': st.selectbox('Tech Support:', ('Yes','No','No internet service')),
            'StreamingTV': st.selectbox('Streaming TV:', ('Yes','No','No internet service')),
            'StreamingMovies': st.selectbox('Streaming Movies:', ('Yes','No','No internet service')),
            'Contract': st.selectbox('Contract:', ('Month-to-month', 'One year', 'Two year')),
            'PaperlessBilling': st.selectbox('Paperless Billing:', ('Yes', 'No')),
            'PaymentMethod': st.selectbox('Payment Method:', (
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)')),
            'MonthlyCharges': st.number_input('Monthly Charges', 0, 150, 70),
            'TotalCharges': st.number_input('Total Charges', 0, 10000, 1000)
        }

        df = pd.DataFrame([inputs])
        st.write("Input Summary:", df)

        if st.button("Predict"):
            try:
                processed = preprocess(df)
                result = model.predict(processed)[0]
                msg = 'Yes, the customer will terminate the service.' if result == 1 else 'No, the customer is happy.'
                st.success(msg)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    else:
        st.subheader("Upload a CSV file for batch prediction")
        uploaded = st.file_uploader("Choose file", type=["csv"])

        if uploaded is not None:
            data = pd.read_csv(uploaded)
            st.write("Data Preview:", data.head())
            if st.button("Predict Batch"):
                try:
                    processed = preprocess(data)
                    results = model.predict(processed)
                    data['Prediction'] = np.where(results == 1, 'Yes', 'No')
                    st.write("Prediction Results:", data[['Prediction']])
                except Exception as e:
                    st.error(f"Error in batch prediction: {e}")

if __name__ == "__main__":
    main()
