import os
 
import joblib
import numpy as np
import streamlit as st
 
# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="centered",
)
 
# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "customer_churn_model.pkl")
 
 
@st.cache_resource
def load_model(path: str):
    """Load the trained model with joblib. Cached so it only loads once."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)
 
 
model = load_model(MODEL_PATH)
 
st.title("Customer Churn Prediction App")
st.write(
    "Enter the customer's details in the sidebar and click **Predict Churn** "
    "to see whether the customer is likely to churn."
)
 
if model is None:
    st.error(
        f"Could not find a model file at `{MODEL_PATH}`. "
        "Make sure `customer_churn_model.pkl` is placed inside a `models/` "
        "folder in the same directory as this app (this path is relative, "
        "so it also works on Streamlit Cloud)."
    )
 
# --------------------------------------------------------------------------
# Sidebar — user inputs
# --------------------------------------------------------------------------
st.sidebar.header("Customer Information")
 
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0
)
total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, max_value=100000.0, value=1000.0, step=10.0
)
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["No", "Yes", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["No", "Yes", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["No", "Yes", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No", "Yes", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No", "Yes", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No", "Yes", "No internet service"]
)
contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
 
predict_button = st.sidebar.button("Predict Churn")
 
 
# --------------------------------------------------------------------------
# Manual encoding helpers
# --------------------------------------------------------------------------
def yes_no(value: str) -> int:
    return 1 if value == "Yes" else 0
 
 
def build_feature_vector() -> np.ndarray:
    """
    Manually encode every input and assemble them into a NumPy array of
    shape (1, 30) matching the get_dummies(drop_first=True) feature order
    described in the module docstring.
    """
    features = [
        senior_citizen,                                            # 0
        tenure,                                                     # 1
        monthly_charges,                                            # 2
        total_charges,                                              # 3
        1 if gender == "Male" else 0,                               # 4
        yes_no(partner),                                            # 5
        yes_no(dependents),                                         # 6
        yes_no(phone_service),                                      # 7
        1 if multiple_lines == "No phone service" else 0,           # 8
        1 if multiple_lines == "Yes" else 0,                        # 9
        1 if internet_service == "Fiber optic" else 0,              # 10
        1 if internet_service == "No" else 0,                       # 11
        1 if online_security == "No internet service" else 0,       # 12
        1 if online_security == "Yes" else 0,                       # 13
        1 if online_backup == "No internet service" else 0,         # 14
        1 if online_backup == "Yes" else 0,                         # 15
        1 if device_protection == "No internet service" else 0,     # 16
        1 if device_protection == "Yes" else 0,                     # 17
        1 if tech_support == "No internet service" else 0,          # 18
        1 if tech_support == "Yes" else 0,                          # 19
        1 if streaming_tv == "No internet service" else 0,          # 20
        1 if streaming_tv == "Yes" else 0,                          # 21
        1 if streaming_movies == "No internet service" else 0,      # 22
        1 if streaming_movies == "Yes" else 0,                      # 23
        1 if contract == "One year" else 0,                         # 24
        1 if contract == "Two year" else 0,                         # 25
        yes_no(paperless_billing),                                  # 26
        1 if payment_method == "Credit card (automatic)" else 0,    # 27
        1 if payment_method == "Electronic check" else 0,           # 28
        1 if payment_method == "Mailed check" else 0,                # 29
    ]
 
    array = np.array(features, dtype=float).reshape(1, -1)
    return array
 
 
# --------------------------------------------------------------------------
# Main area — prediction results
# --------------------------------------------------------------------------
st.subheader("Prediction Result")
 
if predict_button:
    if model is None:
        st.error("Cannot run prediction because the model failed to load.")
    else:
        input_array = build_feature_vector()
 
        if input_array.shape != (1, 30):
            st.error(
                f"Feature vector has shape {input_array.shape}, expected (1, 30). "
                "Check build_feature_vector() against your model's training columns."
            )
        else:
            try:
                prediction = model.predict(input_array)[0]
 
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_array)[0]
 
                if prediction == 1:
                    st.error("⚠️ This customer is likely to **CHURN**.")
                else:
                    st.success("✅ This customer is likely to **STAY**.")
 
                if proba is not None:
                    churn_prob = proba[1] if len(proba) > 1 else proba[0]
                    stay_prob = proba[0] if len(proba) > 1 else 1 - proba[0]
 
                    st.write("### Prediction Probability")
                    col1, col2 = st.columns(2)
                    col1.metric("Stay", f"{stay_prob * 100:.2f}%")
                    col2.metric("Churn", f"{churn_prob * 100:.2f}%")
                    st.progress(float(churn_prob))
 
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Fill in the customer details in the sidebar, then click **Predict Churn**.")
 
st.markdown("---")
st.caption(
    "Model expected at `models/customer_churn_model.pkl` (relative path, "
    "works locally and on Streamlit Cloud)."
)
