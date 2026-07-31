
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# Title and Header
st.title("🛡️ Financial Fraud Detection System")
st.markdown("Enter transaction details below to evaluate the fraud probability.")

# Load the saved model artifacts
@st.cache_resource
def load_model():
    return joblib.load('financial_fraud_detection_model.pkl')

artifacts = load_model()
model = artifacts['model']
scaler = artifacts['scaler']
dummy_cols = artifacts['dummy_columns']
continuous_cols = artifacts['continuous_features']
training_cols = artifacts['training_columns']

# User Inputs
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Hour)", min_value=1, value=1)
    type_val = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

with col2:
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

if st.button("Run Fraud Analysis"):
    # Preprocessing
    input_data = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'balanceChangeOrig': abs(newbalanceOrig - oldbalanceOrg)
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical type
    for col in dummy_cols:
        input_df[col] = 1 if f"type_{type_val}" == col else 0

    # Reorder columns
    input_df = input_df[training_cols]

    # Transform and Scale
    for col in continuous_cols:
        input_df[col] = np.log1p(input_df[col] + 0.001)

    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display Results
    if prediction == 1:
        st.error(f"### 🚨 FRAUDULENT TRANSACTION DETECTED")
        st.write(f"**Fraud Probability:** {probability:.2%}")
    else:
        st.success(f"### ✅ LEGITIMATE TRANSACTION")
        st.write(f"**Fraud Probability:** {probability:.2%}")
