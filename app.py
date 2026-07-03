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
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------------------------
# Global styling (custom CSS)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall app font */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }
 
        /* Header banner */
        .app-header {
            background: linear-gradient(135deg, #1f4e79 0%, #2e86ab 100%);
            padding: 2rem 2rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
        }
        .app-header p {
            margin-top: 0.4rem;
            font-size: 1rem;
            opacity: 0.9;
        }
 
        /* Predict button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #2e86ab 0%, #1f4e79 100%);
            color: white;
            font-weight: 600;
            font-size: 1.05rem;
            padding: 0.7rem 1.4rem;
            border-radius: 10px;
            border: none;
            width: 100%;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(31, 78, 121, 0.35);
            color: white;
        }
 
        /* Result cards */
        .result-card {
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        .result-card.churn {
            background-color: #fdecea;
            border: 1px solid #f5b7b1;
        }
        .result-card.stay {
            background-color: #eafaf1;
            border: 1px solid #a9dfbf;
        }
        .result-card h2 {
            margin: 0 0 0.3rem 0;
        }
 
        /* Confidence badges */
        .badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-low { background-color: #d4efdf; color: #196f3d; }
        .badge-medium { background-color: #fdebd0; color: #9c640c; }
        .badge-high { background-color: #fadbd8; color: #943126; }
 
        /* Footer */
        .app-footer {
            text-align: center;
            padding-top: 1.5rem;
            margin-top: 2rem;
            border-top: 1px solid #eaeaea;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        .app-footer a {
            color: #2e86ab;
            text-decoration: none;
            font-weight: 600;
            margin: 0 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# --------------------------------------------------------------------------
# Model loading (UNCHANGED LOGIC)
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "customer_churn_model.pkl")
 
 
@st.cache_resource
def load_model(path: str):
    """Load the trained model with joblib. Cached so it only loads once."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)
 
 
model = load_model(MODEL_PATH)
 
# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📉 Customer Churn Prediction App</h1>
        <p>An interactive machine learning tool that estimates the likelihood of a
        telecom customer churning, based on their profile, services, and billing
        details. Fill in the customer information and click <b>Predict Churn</b>
        to see the result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
if model is None:
    st.error(
        f"Could not find a model file at `{MODEL_PATH}`. "
        "Make sure `customer_churn_model.pkl` is placed inside a `models/` "
        "folder in the same directory as this app (this path is relative, "
        "so it also works on Streamlit Cloud)."
    )
 
# --------------------------------------------------------------------------
# Sidebar — project information
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ Project Information")
    st.markdown(
        "This app predicts **customer churn** for a telecom company using a "
        "supervised machine learning classification model trained on "
        "historical customer data."
    )
 
    st.markdown("### 🤖 Model Used")
    st.info("Random Forest / Logistic Regression Classifier (Scikit-learn)")
 
    st.markdown("### 🛠️ Technologies")
    st.markdown(
        "- 🐍 **Python**\n"
        "- 🎈 **Streamlit**\n"
        "- 📊 **Scikit-learn**\n"
        "- 🔢 **NumPy**"
    )
 
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Your Name Here**")
 
    st.markdown("---")
    st.caption(
        "Model expected at `models/customer_churn_model.pkl` "
        "(relative path — works locally and on Streamlit Cloud)."
    )
 
# --------------------------------------------------------------------------
# Main area — input form, organized into sections
# --------------------------------------------------------------------------
st.markdown("### 📝 Customer Details")
 
# --- Section 1: Demographics -----------------------------------------------
with st.expander("👤 Demographics", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female"], help="Customer's gender"
        )
    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            help="Is the customer a senior citizen? 0 = No, 1 = Yes",
        )
    with col3:
        partner = st.selectbox(
            "Partner", ["Yes", "No"], help="Does the customer have a partner?"
        )
    with col4:
        dependents = st.selectbox(
            "Dependents", ["Yes", "No"], help="Does the customer have dependents?"
        )
 
# --- Section 2: Account & Billing ------------------------------------------
with st.expander("💳 Account & Billing", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            help="Number of months the customer has stayed with the company",
        )
        monthly_charges = st.slider(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            help="Amount charged to the customer each month",
        )
    with col2:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            help="Does the customer use paperless billing?",
        )
        total_charges = st.slider(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0,
            help="Total amount charged to the customer to date",
        )
 
    col3, col4 = st.columns(2)
    with col3:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="The customer's current contract term",
        )
    with col4:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            help="How the customer pays their bill",
        )
 
# --- Section 3: Phone & Internet Services -----------------------------------
with st.expander("📡 Phone & Internet Services", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"], help="Does the customer have phone service?"
        )
    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            help="Does the customer have multiple phone lines?",
        )
    with col3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            help="Customer's internet service provider type",
        )
 
# --- Section 4: Add-on Services ---------------------------------------------
with st.expander("🧰 Add-on Services", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            help="Does the customer have online security add-on?",
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
            help="Does the customer have online backup add-on?",
        )
    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
            help="Does the customer have device protection add-on?",
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            help="Does the customer have tech support add-on?",
        )
    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            help="Does the customer have streaming TV add-on?",
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            help="Does the customer have streaming movies add-on?",
        )
 
st.write("")
predict_button = st.button("🔮 Predict Churn", use_container_width=True)
 
 
# --------------------------------------------------------------------------
# Manual encoding helpers (UNCHANGED LOGIC)
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
 
 
def get_confidence_badge(probability: float) -> str:
    """
    Return an HTML badge representing risk confidence based on the churn
    probability. This is purely a presentational helper and does not
    affect the prediction logic.
    """
    if probability < 0.4:
        return '<span class="badge badge-low">🟢 Low Risk</span>'
    elif probability < 0.7:
        return '<span class="badge badge-medium">🟡 Medium Risk</span>'
    else:
        return '<span class="badge badge-high">🔴 High Risk</span>'
 
 
# --------------------------------------------------------------------------
# Main area — prediction results
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Prediction Result")
 
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
 
                if proba is not None:
                    churn_prob = proba[1] if len(proba) > 1 else proba[0]
                    stay_prob = proba[0] if len(proba) > 1 else 1 - proba[0]
                else:
                    churn_prob = float(prediction)
                    stay_prob = 1 - churn_prob
 
                # ---- Result card -------------------------------------------------
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card churn">
                            <h2>⚠️ This customer is likely to <span style="color:#c0392b;">CHURN</span></h2>
                            <p>Consider proactive retention actions such as personalized
                            offers, loyalty discounts, or a check-in call.</p>
                            {get_confidence_badge(churn_prob)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card stay">
                            <h2>✅ This customer is likely to <span style="color:#1e8449;">STAY</span></h2>
                            <p>This customer shows a healthy engagement profile with the
                            current services and contract.</p>
                            {get_confidence_badge(churn_prob)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
 
                # ---- Probability breakdown ----------------------------------------
                st.markdown("#### 📈 Prediction Probability")
                col1, col2 = st.columns(2)
                col1.metric("Stay Probability", f"{stay_prob * 100:.2f}%")
                col2.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
 
                st.write("Churn likelihood:")
                st.progress(float(churn_prob))
 
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("👈 Fill in the customer details above, then click **Predict Churn** to see the result.")
 
# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built with ❤️ using Streamlit &amp; Scikit-learn<br>
        <a href="https://github.com/your-username" target="_blank">🐙 GitHub</a>
        |
        <a href="https://www.linkedin.com/in/your-profile" target="_blank">💼 LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
