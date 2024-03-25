import pickle
import numpy as np
import streamlit as st

# Function to load the model from disk
def load_model(filename='finalized_xgboost_model.sav'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict the probability of diabetes and categorize the risk
def predict_diabetes_risk_category(model, input_data):
    probabilities = model.predict_proba(input_data)
    diabetes_probability = probabilities[:, 1][0]  # Probability of '1' (Diabetes)

    # Recommendations based on risk category
    low_risk_advice = """
    For Low Risk Individuals:
    - Healthy Diet: Focus on a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.
    - Regular Exercise: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.
    - Weight Management: Maintain a healthy weight to reduce the risk of diabetes.
    - Regular Check-ups: Have regular physical exams and blood tests as recommended.
    """
    
    medium_risk_advice = """
    For Medium Risk Individuals:
    - Increased Monitoring: More frequent blood glucose screenings may be recommended.
    - Lifestyle Interventions: Participate in a lifestyle intervention program focusing on dietary changes and physical activity.
    - Manage Other Health Conditions: Control risk factors such as high blood pressure and high cholesterol.
    """
    
    high_risk_advice = """
    For High Risk Individuals:
    - Medical Consultation: Regularly consult with a healthcare provider to closely monitor your health status.
    - Medications: May prescribe medications to manage your blood sugar levels.
    - Structured Diet and Exercise Programs: Engage in a structured lifestyle program to prevent or delay type 2 diabetes.
    - Weight Loss Surgery: Considered for those who are significantly overweight.
    """

    if diabetes_probability < 0.1:
        risk_category = "Low Risk of having diabetes"
        advice = low_risk_advice
    elif diabetes_probability < 0.35:
        risk_category = "Medium Risk of Having Diabetes"
        advice = medium_risk_advice
    else:
        risk_category = "High Risk of having Diabetes"
        advice = high_risk_advice
    
    return diabetes_probability, risk_category, advice

# Load the previously saved XGBoost model
loaded_model = load_model()

# Streamlit user interface for input
st.title('Diabetes Risk Prediction App')

# User input fields
gender = st.selectbox('Gender', ['Male', 'Female'], index=0)
age = st.slider('Age', 0, 100, 25)
hypertension = st.radio('Hypertension', ['No', 'Yes'], index=0)
heart_disease = st.radio('Heart Disease', ['No', 'Yes'], index=0)
smoking_history = st.selectbox('Smoking History', ['non-smoker', 'past-smoker', 'current-smoker'], index=0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)
hba1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=400, value=100)

# Encoding the inputs to match training data encoding
gender_encoded = 1 if gender == 'Female' else 0
hypertension_encoded = 1 if hypertension == 'Yes' else 0
heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
smoking_history_encoded = {'non-smoker': 1, 'past-smoker': 2, 'current-smoker': 0}[smoking_history]

# Preparing the input array for the model
input_features = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, smoking_history_encoded, bmi, hba1c_level, blood_glucose_level]])

# Predict button and displaying the categorized risk and advice
if st.button('Predict Risk of Diabetes'):
    probability, risk_category, advice = predict_diabetes_risk_category(loaded_model, input_features)
    st.write(f'The probability of having diabetes is: {probability:.2f}')
    st.write(risk_category)
    st.write(advice)