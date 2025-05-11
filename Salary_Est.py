import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load Keras model
model = load_model("salary_model.h5")

# Load scaler and model columns
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Streamlit UI
st.title("💼 Salary Prediction App")

education = st.selectbox("Select Education", ['Bachelors', 'Masters', 'PhD'])
location = st.selectbox("Select Location", ['New York', 'San Francisco', 'Chicago'])
job_title = st.selectbox("Select Job Title", ['Data Scientist', 'Software Engineer', 'Analyst'])
experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)

# Prepare input
input_df = pd.DataFrame([{
    'Education': education,
    'Location': location,
    'Job_Title': job_title,
    'Experience': experience
}])

# One-hot encode
input_encoded = pd.get_dummies(input_df)

# Align with training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# Predict
prediction = model.predict(input_scaled)[0][0]  # output is [[value]], so [0][0]

# Show prediction
st.success(f"💰 Estimated Salary: ${prediction:.2f}")




