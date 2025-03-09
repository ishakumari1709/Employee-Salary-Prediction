import streamlit as st
import joblib
import numpy as np

# Load the model and encoders
model = joblib.load('salary_prediction_model.pkl')
degree_encoder = joblib.load('degree_encoder.pkl')
job_title_encoder = joblib.load('job_title_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App Title
st.title("💼 Salary Prediction App")

# Input Fields
age = st.number_input("Enter Age:", min_value=18, max_value=70, step=1)
gender = st.radio("Select Gender:", ['Male', 'Female'])
degree = st.selectbox("Select Degree:", degree_encoder.classes_)
job_title = st.selectbox("Select Job Title:", job_title_encoder.classes_)
experience_years = st.number_input("Enter Experience (Years):", min_value=0, max_value=50, step=1)

# Predict Salary Button
if st.button("Predict Salary"):
    try:
        # Gender Conversion
        gender = 0 if gender == 'Male' else 1

        # Scaling and Encoding
        age_scaled = scaler.transform([[age]])[0][0]
        experience_scaled = scaler.transform([[experience_years]])[0][0]

        # Handle unseen labels by adding new categories if needed
        if degree not in degree_encoder.classes_:
            degree_encoder.classes_ = np.append(degree_encoder.classes_, degree)
        degree_encoded = degree_encoder.transform([degree])[0]

        if job_title not in job_title_encoder.classes_:
            job_title_encoder.classes_ = np.append(job_title_encoder.classes_, job_title)
        job_title_encoded = job_title_encoder.transform([job_title])[0]

        # Prediction
        prediction = model.predict([[age_scaled, gender, degree_encoded, job_title_encoded, experience_scaled]])

        st.success(f'💰 Predicted Salary: **${prediction[0]:,.2f}**')
    except Exception as e:
        st.error(f"❗ Error: {e}")
