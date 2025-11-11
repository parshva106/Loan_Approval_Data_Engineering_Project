import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high contrast and modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px #000000;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #00D4AA;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .prediction-approved {
        background: linear-gradient(135deg, #00D4AA, #00A082);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 900;
        margin: 2rem 0;
        border: 3px solid #00FFCC;
        box-shadow: 0 8px 16px rgba(0, 212, 170, 0.3);
    }
    .prediction-rejected {
        background: linear-gradient(135deg, #FF4B4B, #C13535);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 900;
        margin: 2rem 0;
        border: 3px solid #FF6B6B;
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(106, 17, 203, 0.4);
    }
    .feature-card {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #4A235A 100%);
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model and preprocessors"""
    try:
        with open('loan_approval_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        # Debug: Check what's in the pickle file
        st.sidebar.write("🔍 Debug Info:")
        st.sidebar.write(f"Type: {type(model_data)}")
        
        if isinstance(model_data, tuple):
            st.sidebar.write(f"Length: {len(model_data)}")
            for i, item in enumerate(model_data):
                st.sidebar.write(f"Item {i}: {type(item)}")
        
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_components(model_data):
    """Extract model and preprocessors from the loaded data"""
    components = {
        'model': None,
        'scaler': None,
        'label_encoders': {}
    }
    
    if isinstance(model_data, tuple):
        # If it's a tuple, iterate through items to find components
        for item in model_data:
            if isinstance(item, DecisionTreeClassifier):
                components['model'] = item
                st.sidebar.success("✅ Decision Tree Model Found")
            elif isinstance(item, StandardScaler):
                components['scaler'] = item
                st.sidebar.success("✅ Standard Scaler Found")
            elif isinstance(item, LabelEncoder):
                # For LabelEncoders, we need to identify which one it is
                # This is a simplified approach - you might need to adjust based on your data
                if hasattr(item, 'classes_'):
                    if 'Male' in item.classes_:
                        components['label_encoders']['Gender'] = item
                        st.sidebar.success("✅ Gender Encoder Found")
                    elif 'Yes' in item.classes_:
                        components['label_encoders']['Married'] = item
                        st.sidebar.success("✅ Married Encoder Found")
                    elif '3+' in item.classes_:
                        components['label_encoders']['Dependents'] = item
                        st.sidebar.success("✅ Dependents Encoder Found")
                    elif 'Graduate' in item.classes_:
                        components['label_encoders']['Education'] = item
                        st.sidebar.success("✅ Education Encoder Found")
                    elif 'Urban' in item.classes_:
                        components['label_encoders']['Property_Area'] = item
                        st.sidebar.success("✅ Property Area Encoder Found")
                    elif 'Y' in item.classes_:
                        components['label_encoders']['Loan_Status'] = item
                        st.sidebar.success("✅ Loan Status Encoder Found")
    
    return components

def preprocess_input(data, components):
    """Preprocess the input data using the loaded components"""
    try:
        # Create mapping dictionaries for categorical variables
        gender_map = {'Male': 1, 'Female': 0}
        married_map = {'Yes': 1, 'No': 0}
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_map = {'Yes': 1, 'No': 0}
        dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
        property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        
        # Apply manual encoding (fallback if label encoders aren't available)
        processed_data = {
            'Gender': gender_map[data['Gender']],
            'Married': married_map[data['Married']],
            'Dependents': dependents_map[data['Dependents']],
            'Education': education_map[data['Education']],
            'Self_Employed': self_employed_map[data['Self_Employed']],
            'ApplicantIncome': data['ApplicantIncome'],
            'CoapplicantIncome': data['CoapplicantIncome'],
            'LoanAmount': data['LoanAmount'],
            'Loan_Amount_Term': data['Loan_Amount_Term'],
            'Credit_History': data['Credit_History'],
            'Property_Area': property_area_map[data['Property_Area']]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([processed_data])
        
        # Apply scaling if scaler is available
        if components['scaler'] is not None:
            numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
            input_df[numerical_columns] = components['scaler'].transform(input_df[numerical_columns])
        
        return input_df
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def make_prediction(model, input_data):
    """Make prediction using the model"""
    try:
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        return prediction[0], probabilities[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">🏦 SMART LOAN APPROVAL PREDICTOR</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.error("❌ Failed to load model. Please check your model file.")
        return
    
    # Extract components
    components = extract_components(model_data)
    
    if components['model'] is None:
        st.warning("⚠️ Model not found in the loaded data. Using demo mode.")
        demo_mode = True
    else:
        demo_mode = False
        st.sidebar.success("🎯 Model loaded successfully!")
    
    # Sidebar for user input
    with st.sidebar:
        st.markdown("### 🔧 Application Settings")
        st.markdown("---")
        
        st.markdown("### 📊 Input Features")
        st.markdown("Fill in your details below:")
        
        # User inputs
        applicant_income = st.slider("💰 Applicant Income ($)", 1500, 81000, 5000, 
                                   help="Monthly income of the applicant")
        coapplicant_income = st.slider("💼 Co-applicant Income ($)", 0, 41000, 0,
                                     help="Monthly income of co-applicant")
        loan_amount = st.slider("🏠 Loan Amount ($)", 9000, 700000, 100000, step=1000,
                              help="Requested loan amount")
        loan_term = st.slider("⏰ Loan Term (months)", 12, 480, 360,
                            help="Loan repayment period")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👤 Gender", ["Male", "Female"])
            married = st.selectbox("💍 Married", ["No", "Yes"])
            education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("💼 Self Employed", ["No", "Yes"])
        
        with col2:
            dependents = st.selectbox("👨‍👩‍👧‍👦 Dependents", ["0", "1", "2", "3+"])
            credit_history = st.selectbox("📈 Credit History", [1, 0], 
                                       format_func=lambda x: "Good" if x == 1 else "Needs Improvement")
            property_area = st.selectbox("🏡 Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📋 Application Summary</div>', 
                   unsafe_allow_html=True)
        
        # Display input summary in cards
        col1a, col2a = st.columns(2)
        
        with col1a:
            st.markdown(f"""
            <div class="feature-card">
                <strong>Personal Information</strong><br>
                Gender: {gender}<br>
                Married: {married}<br>
                Dependents: {dependents}<br>
                Education: {education}<br>
                Self Employed: {self_employed}
            </div>
            """, unsafe_allow_html=True)
            
        with col2a:
            st.markdown(f"""
            <div class="feature-card">
                <strong>Financial Information</strong><br>
                Applicant Income: ${applicant_income:,}<br>
                Co-applicant Income: ${coapplicant_income:,}<br>
                Loan Amount: ${loan_amount:,}<br>
                Loan Term: {loan_term} months
            </div>
            """, unsafe_allow_html=True)
        
        # Additional info card
        st.markdown(f"""
        <div class="feature-card">
            <strong>Loan Details</strong><br>
            Credit History: {"Good" if credit_history == 1 else "Needs Improvement"}<br>
            Property Area: {property_area}<br>
            Debt-to-Income Ratio: {(loan_amount/loan_term)/applicant_income:.1%}
        </div>
        """, unsafe_allow_html=True)
            
        # Risk assessment visualization
        st.markdown("### 📊 Risk Assessment")
        
        # Calculate a simple risk score
        risk_score = 0
        if credit_history == 0:
            risk_score += 40
        if applicant_income < 3000:
            risk_score += 20
        if coapplicant_income == 0:
            risk_score += 10
        if loan_amount > 300000:
            risk_score += 15
        if education == "Not Graduate":
            risk_score += 10
        if self_employed == "Yes":
            risk_score += 5
            
        risk_level = "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High"
        risk_color = "#00D4AA" if risk_level == "Low" else "#FFA500" if risk_level == "Medium" else "#FF4B4B"
        
        # Risk gauge
        st.markdown(f"""
        <div style="background: #1E1E1E; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <div style="background: linear-gradient(90deg, #00D4AA 0%, #FFA500 50%, #FF4B4B 100%); 
                        height: 20px; border-radius: 10px; margin-bottom: 10px; position: relative;">
                <div style="position: absolute; left: {risk_score}%; top: -5px; 
                            width: 4px; height: 30px; background: white; border-radius: 2px;"></div>
            </div>
            <strong style="color: {risk_color}; font-size: 1.2rem;">Risk Level: {risk_level}</strong><br>
            <span style="color: #666;">Score: {risk_score}/100</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction</div>', 
                   unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🚀 PREDICT LOAN APPROVAL", use_container_width=True):
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            with st.spinner('🔍 Analyzing your application...'):
                time.sleep(2)  # Simulate processing time
                
                if demo_mode:
                    # Demo prediction logic
                    if (credit_history == 1 and applicant_income > 2500 and 
                        (loan_amount/loan_term)/applicant_income < 0.5):
                        prediction = 1  # Approved
                        probability = 0.82
                    else:
                        prediction = 0  # Rejected
                        probability = 0.76
                else:
                    # Real model prediction
                    processed_data = preprocess_input(input_data, components)
                    if processed_data is not None:
                        prediction, proba = make_prediction(components['model'], processed_data)
                        if prediction is not None:
                            probability = max(proba)
                        else:
                            # Fallback to demo logic if prediction fails
                            st.warning("Using fallback prediction logic")
                            if (credit_history == 1 and applicant_income > 2500):
                                prediction = 1
                                probability = 0.75
                            else:
                                prediction = 0
                                probability = 0.70
                    else:
                        st.error("Preprocessing failed")
                        return
            
            # Display result
            st.markdown("### 🎯 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-approved">
                    ✅ APPROVED!<br>
                    <small>Confidence: {probability:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                
                st.markdown("""
                ### 🎉 Next Steps:
                - Our team will contact you within 24 hours
                - Prepare your documents for verification
                - Final approval subject to document verification
                """)
            else:
                st.markdown(f"""
                <div class="prediction-rejected">
                    ❌ NOT APPROVED<br>
                    <small>Confidence: {probability:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                ### 💡 Suggestions for Improvement:
                - Improve your credit score
                - Consider reducing loan amount
                - Add a co-applicant with stable income
                - Provide additional collateral
                - Increase your income stability
                """)
        
        # Statistics section
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.metric("Approval Rate", "68%", "2%")
            st.metric("Avg Processing", "3-5 days")
        
        with col_stat2:
            st.metric("Interest Rate", "7.5-12%")
            st.metric("Satisfaction", "92%", "3%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>Disclaimer:</strong> This is a predictive model for demonstration purposes. 
        Actual loan approval is subject to bank's internal policies and verification processes.
        <br>Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()