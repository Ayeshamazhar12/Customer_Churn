import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/customer_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title(" Customer Churn Prediction App")
st.write("Predict whether a customer will leave the company or stay.")

# Sidebar Inputs
st.sidebar.header("Customer Input Features")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
TotalCharges = st.sidebar.slider("Total Charges", 0, 10000, 1000)

MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# Convert input into model format (30 features)

def encode_input():
    input_data = np.zeros(30)

    input_data[0] = 1 if gender == "Male" else 0
    input_data[1] = SeniorCitizen
    input_data[2] = 1 if Partner == "Yes" else 0
    input_data[3] = 1 if Dependents == "Yes" else 0
    input_data[4] = tenure
    input_data[5] = 1 if PhoneService == "Yes" else 0
    input_data[6] = 1 if PaperlessBilling == "Yes" else 0
    input_data[7] = MonthlyCharges
    input_data[8] = TotalCharges

    # MultipleLines
    input_data[9] = 1 if MultipleLines == "No phone service" else 0
    input_data[10] = 1 if MultipleLines == "Yes" else 0

    # InternetService
    input_data[11] = 1 if InternetService == "Fiber optic" else 0
    input_data[12] = 1 if InternetService == "No" else 0

    # OnlineSecurity
    input_data[13] = 1 if OnlineSecurity == "No internet service" else 0
    input_data[14] = 1 if OnlineSecurity == "Yes" else 0

    # OnlineBackup
    input_data[15] = 1 if OnlineBackup == "No internet service" else 0
    input_data[16] = 1 if OnlineBackup == "Yes" else 0

    # DeviceProtection
    input_data[17] = 1 if DeviceProtection == "No internet service" else 0
    input_data[18] = 1 if DeviceProtection == "Yes" else 0

    # TechSupport
    input_data[19] = 1 if TechSupport == "No internet service" else 0
    input_data[20] = 1 if TechSupport == "Yes" else 0

    # StreamingTV
    input_data[21] = 1 if StreamingTV == "No internet service" else 0
    input_data[22] = 1 if StreamingTV == "Yes" else 0

    # StreamingMovies
    input_data[23] = 1 if StreamingMovies == "No internet service" else 0
    input_data[24] = 1 if StreamingMovies == "Yes" else 0

    # Contract
    input_data[25] = 1 if Contract == "One year" else 0
    input_data[26] = 1 if Contract == "Two year" else 0

    # PaymentMethod
    input_data[27] = 1 if PaymentMethod == "Credit card (automatic)" else 0
    input_data[28] = 1 if PaymentMethod == "Electronic check" else 0
    input_data[29] = 1 if PaymentMethod == "Mailed check" else 0

    return input_data.reshape(1, -1)


if st.button(" Predict Churn"):
    input_data = encode_input()

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f" Customer will CHURN")
    else:
        st.success(f" Customer will STAY")

    st.metric("Churn Probability", f"{probability:.2f}")
