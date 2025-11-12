# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Card-like containers */
    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #00c853;
        color: white;
        font-size: 1rem;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #00e676;
        color: black;
        transform: scale(1.05);
    }
    /* Titles */
    h1, h2, h3 {
        color: #00e5ff;
        text-shadow: 0px 0px 10px #00e5ff;
    }
    /* Center text */
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
try:
    with open("loan_approval_model.pkl", "rb") as f:
        model, scaler, encoders = pickle.load(f)
except Exception as e:
    st.error("⚠️ Error loading model: " + str(e))
    st.stop()

# ---- APP HEADER ----
st.markdown("<h1 class='center-text'>🏦 Loan Approval Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text'>Easily predict whether your loan will be approved using machine learning 💡</p>", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/833/833314.png", width=100)
st.sidebar.markdown("## Enter Applicant Details")

# ---- USER INPUTS ----
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.sidebar.selectbox("Loan Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
credit_history = st.sidebar.selectbox("Credit History", [0, 1])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---- INPUT DATAFRAME ----
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

# ---- PREPROCESS INPUT ----
for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

df_input_scaled = scaler.transform(df_input)

# ---- MAIN CONTENT ----
st.markdown("### 📋 Applicant Summary")
st.dataframe(df_input, use_container_width=True)

# ---- PREDICTION ----
if st.button("🔮 Predict Loan Approval"):
    with st.spinner("Analyzing application... ⏳"):
        prediction = model.predict(df_input_scaled)[0]
        prob = None
        try:
            prob = model.predict_proba(df_input_scaled)[0][1]
        except:
            prob = None

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown("<h2 class='center-text'>✅ Loan Approved!</h2>", unsafe_allow_html=True)
        st.balloons()
        st.markdown("<p class='center-text'>Congratulations! Based on your details, your loan application is likely to be approved. 🎉</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='center-text'>❌ Loan Rejected</h2>", unsafe_allow_html=True)
        st.markdown("<p class='center-text'>Unfortunately, your loan application might not be approved. Consider improving your credit history or income.</p>", unsafe_allow_html=True)

    if prob is not None:
        st.progress(float(prob))
        st.markdown(f"<p class='center-text'>Approval Probability: <b>{prob*100:.2f}%</b></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
---
<div class='center-text'>
    <p>Made with ❤️ using <b>Streamlit</b> | Machine Learning Powered</p>
</div>
""", unsafe_allow_html=True)
