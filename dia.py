import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load your dataset
data = pd.read_csv('mlp.csv')

# Load pre-trained model
model = load_model('diabetes_prediction_model.h5')

# Streamlit UI
st.title('Diabetes Prediction')

# Input fields for user to enter data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20)
glucose = st.number_input('Glucose', min_value=0, max_value=500)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
insulin = st.number_input('Insulin', min_value=0, max_value=1000)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0,  step=0.1)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0,  step=0.01)
age = st.number_input('Age', min_value=0, max_value=120)

if st.button('Predict'):
    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age],
    })

    # Merge user input with dataset columns (excluding target 'Outcome' column)
    X = data.drop('Outcome', axis=1)
    X = pd.concat([X, user_input], ignore_index=True)

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retain user input data for prediction
    user_input_scaled = X_scaled[-1].reshape(1, -1)
    X_scaled = X_scaled[:-1]

    # Make predictions on the user input
    prediction = model.predict(user_input_scaled)
    result = 'Diabetic' if prediction[0][0] >= 0.5 else 'Non-Diabetic'

    # Display prediction result
    st.write(f'Prediction: {result}')
