import pickle
import numpy as np
import streamlit as st
import random

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
    <span style='color: green; font-weight: bold; font-size: 24px; line-height: 1.6;'>
    For Low Risk Individuals:
    <li>Healthy Diet: Focus on a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.</li>
    <li>Regular Exercise: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.</li>
    <li>Weight Management: Maintain a healthy weight to reduce the risk of diabetes.</li>
    <li>Regular Check-ups: Have regular physical exams and blood tests as recommended.</li>
    </span>
    """
    
    medium_risk_advice = """
    <span style='color: yellow; font-weight: bold; font-size: 24px; line-height: 1.6;'>
    For Medium Risk Individuals:
    <li>Increased Monitoring: More frequent blood glucose screenings may be recommended.</li>
    <li>Lifestyle Interventions: Participate in a lifestyle intervention program focusing on dietary changes and physical activity.</li>
    <li>Manage Other Health Conditions: Control risk factors such as high blood pressure and high cholesterol.</li>
    </span>
    """
    
    high_risk_advice = """
    <span style='color: red; font-weight: bold; font-size: 24px; line-height: 1.6;'>
    For High Risk Individuals:
    <li>Medical Consultation: Regularly consult with a healthcare provider to closely monitor your health status.</li>
    <li>Medications: May prescribe medications to manage your blood sugar levels.</li>
    <li>Structured Diet and Exercise Programs: Engage in a structured lifestyle program to prevent or delay type 2 diabetes.</li>
    <li>Weight Loss Surgery: Considered for those who are significantly overweight.</li>
    </span>
    """

    if diabetes_probability < 0.2:
        risk_category = "Low Risk of having diabetes"
        advice = low_risk_advice
    elif diabetes_probability < 0.8:
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

# Sidebar for project and team member names
st.sidebar.title('App Developed By:')
#st.sidebar.write('TEAM MEMBERS')
team_members = ['Falguni Bharadwaj']
random.shuffle(team_members)  # Shuffling the team member names
st.sidebar.write('Team Members:')
for member in team_members:
    st.sidebar.write(member)
    
# Download link for the standalone app
url = "https://mailuc-my.sharepoint.com/personal/bharadfi_mail_uc_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fbharadfi%5Fmail%5Fuc%5Fedu%2FDocuments%2FDiabetes%20Risk%20Predictor%2Ezip"
st.markdown(
    f"<a href='{url}' target='_blank'>"
    "<img src='https://logodownload.org/wp-content/uploads/2022/07/microsoft-store-logo-0-2048x2048.png' "
    "alt='Microsoft Store Download' style='width:50px; height:auto; margin-top:10px;'></a>"
    "<div style='margin-top: 5px;'>Note: Click on the Microsoft Store logo above to download standlone Windows (x86/x64) app. Just extract and run 'diabetes_risk_predictor.exe' with active internet connection and without any proxy settings</div>",
    unsafe_allow_html=True
)

# User input fields
gender = st.selectbox('Gender', ['Male', 'Female'], index=1)
age = st.slider('Age', 0, 100, 54)
hypertension = st.radio('Hypertension', ['No', 'Yes'], index=1)
heart_disease = st.radio('Heart Disease', ['No', 'Yes'], index=1)
smoking_history = st.selectbox('Smoking History', ['non-smoker', 'past-smoker', 'current-smoker'], index=2)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=24.5)
hba1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=6.76)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=400, value=189)

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
    st.markdown(f"<h2 style='font-weight: bold; font-size: 30px;'>{risk_category}</h2>", unsafe_allow_html=True)
    st.markdown(advice, unsafe_allow_html=True)
