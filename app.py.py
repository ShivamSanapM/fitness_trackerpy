import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from streamlit_option_menu import option_menu

# Set Page Config
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Hide Default Streamlit Style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load Models
models = {
    'diabetes': pickle.load(open('SavedModels/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('SavedModels/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('SavedModels/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('SavedModels/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('SavedModels/Thyroid_model.sav', 'rb'))
}

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4319/4319392.png", width=200)
    st.title("⚕️ Disease Prediction")
    selected = option_menu(
        "Select a Disease",
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction',
         'Lung Cancer Prediction',
         'Hypo-Thyroid Prediction'],
        icons=['droplet', 'heart', 'brain', 'lungs', 'thermometer'],
        menu_icon="stethoscope",
        default_index=0
    )

# Function for input display
def display_input(label, tooltip, key, type="text"):
    return st.number_input(label, key=key, help=tooltip, step=0.1 if type == "float" else 1)

# Prediction Function
def predict(model, features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return prediction[0], probability

# Function to generate a downloadable report
def create_download_link(prediction_text):
    b64 = base64.b64encode(prediction_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="prediction_result.txt">Download Report</a>'
    return href

# **Diabetes Prediction**
if selected == 'Diabetes Prediction':
    st.title("🩸 Diabetes Risk Assessment")
    st.subheader("Enter Your Details:")
    
    Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies')
    Glucose = display_input('Glucose Level', 'Enter glucose level', 'Glucose')
    BloodPressure = display_input('Blood Pressure', 'Enter blood pressure', 'BloodPressure')
    SkinThickness = display_input('Skin Thickness', 'Enter skin thickness', 'SkinThickness')
    Insulin = display_input('Insulin Level', 'Enter insulin level', 'Insulin')
    BMI = display_input('BMI', 'Enter Body Mass Index', 'BMI', 'float')
    DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter pedigree function value', 'DPF', 'float')
    Age = display_input('Age', 'Enter your age', 'Age')

    if st.button('🔍 Check Diabetes Risk'):
        prediction, probability = predict(models['diabetes'], [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        result_text = '🚨 You are at risk for Diabetes.' if prediction == 1 else '✅ No signs of Diabetes detected.'
        st.success(result_text)
        
        if probability is not None:
            st.write(f"Confidence: **{round(max(probability[0]) * 100, 2)}%**")
        
        st.markdown(create_download_link(result_text), unsafe_allow_html=True)

# **Heart Disease Prediction**
if selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Assessment")
    st.subheader("Enter Your Health Data:")
    
    age = display_input('Age', 'Enter your age', 'age')
    sex = display_input('Sex (1 = Male, 0 = Female)', 'Enter your sex', 'sex')
    cp = display_input('Chest Pain Type (0-3)', 'Enter chest pain type', 'cp')
    trestbps = display_input('Resting Blood Pressure', 'Enter resting blood pressure', 'trestbps')
    chol = display_input('Cholesterol Level', 'Enter cholesterol level', 'chol')
    fbs = display_input('Fasting Blood Sugar (>120 mg/dl: 1, else 0)', 'Enter fasting blood sugar', 'fbs')
    restecg = display_input('Resting ECG Results (0-2)', 'Enter ECG results', 'restecg')
    thalach = display_input('Max Heart Rate Achieved', 'Enter max heart rate', 'thalach')
    exang = display_input('Exercise Induced Angina (1 = Yes, 0 = No)', 'Enter angina status', 'exang')
    oldpeak = display_input('ST Depression Induced by Exercise', 'Enter ST depression value', 'oldpeak', 'float')
    slope = display_input('Slope of Peak Exercise ST Segment (0-2)', 'Enter slope value', 'slope')
    ca = display_input('Major Vessels Colored by Fluoroscopy (0-3)', 'Enter number of vessels', 'ca')
    thal = display_input('Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)', 'Enter thal value', 'thal')

    if st.button('🔍 Check Heart Disease Risk'):
        prediction, probability = predict(models['heart_disease'], [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        result_text = '🚨 You might have Heart Disease.' if prediction == 1 else '✅ No signs of Heart Disease detected.'
        st.success(result_text)
        
        if probability is not None:
            st.write(f"Confidence: **{round(max(probability[0]) * 100, 2)}%**")
        
        st.markdown(create_download_link(result_text), unsafe_allow_html=True)

# **Parkinson's Prediction, Lung Cancer, and Thyroid can be implemented similarly with enhancements!**
