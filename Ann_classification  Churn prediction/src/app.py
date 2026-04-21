import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '..')

# Load model
model = tf.keras.models.load_model(os.path.join(PROJECT_DIR, 'models', 'model.h5'), compile=False)

# Load encoders
with open(os.path.join(PROJECT_DIR, 'encoders', 'scaler.pkl'), 'rb') as f:
    scalar = pickle.load(f)
with open(os.path.join(PROJECT_DIR, 'encoders', 'one_hot_encoder_geography.pkl'), 'rb') as f:
    onehotencoder = pickle.load(f)
with open(os.path.join(PROJECT_DIR, 'encoders', 'label_encoder_gender.pkl'), 'rb') as f:
    labelencoder = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction App")

# Input data from user
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', labelencoder.classes_)
age = st.slider('Age', 18, 100, 30)
tenure = st.slider('Tenure', 0, 10, 3)
balance = st.number_input('Balance')
creditscore = st.number_input('Credit Score')
estimatedsalary = st.number_input('Estimated Salary')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [labelencoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimatedsalary],
})

# One-hot encode geography
geo_input = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehotencoder.transform(geo_input).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

# Combine and reorder to match training column order
input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_data = input_data[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                          'Geography_France', 'Geography_Germany', 'Geography_Spain']]
input_data_scaled = scalar.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f"Predicted probability of churn: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
