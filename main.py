import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the label encoders
label_encoders = {}
for column in ['Gender', 'Marital Status', 'Employment Status', 'Location (Urban/Rural)', 'Tier']:
    filename = f'{column.replace("/", "_")}_encoder.pkl'  # Replace '/' with '_'
    label_encoders[column] = joblib.load(filename)


def preprocess_input_data(input_data):
    # Encode categorical variables
    for column, encoder in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(input_data[column])
    # Standardize features
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled


def predict_default_status(input_data):
    input_data_scaled = preprocess_input_data(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction


def main():
    st.title('Default Status Prediction')
    st.write('Enter the details to predict default status:')

    # Create input fields for user input
    age = st.number_input('Age', min_value=0, max_value=150, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
    employment_status = st.selectbox('Employment Status', ['Employed', 'Self-employed', 'Unemployed'])
    monthly_income = st.number_input('Monthly Income (USD)', min_value=0)
    location = st.selectbox('Location (Urban/Rural)', ['Urban', 'Rural'])
    credit_score = st.number_input('Credit Score', min_value=0)
    tier = st.selectbox('Tier', ['Bronze', 'Silver', 'Gold', 'Platinum'])
    family = st.number_input('Family', min_value=0)
    dependants = st.number_input('Dependants', min_value=0)
    number_of_dependants = st.number_input('Number of Dependents', min_value=0)
    total_dependents = st.number_input('Total Dependents', min_value=0)
    monthly_premium = st.number_input('Monthly Premium', min_value=0)
    monthly_inflation_rate = st.number_input('Monthly Inflation Rate (%)', min_value=0)

    # Create a DataFrame with user input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Employment Status': [employment_status],
        'Monthly Income (USD)': [monthly_income],
        'Location (Urban/Rural)': [location],
        'Credit Score': [credit_score],
        'Tier': [tier],
        'Family': [family],
        'Dependants': [dependants],
        'Number of Dependants': [number_of_dependants],
        'Total Dependents': [total_dependents],
        'Monthly Premium': [monthly_premium],
        'Monthly Inflation Rate (%)': [monthly_inflation_rate]
    })

    if st.button('Predict'):
        prediction = predict_default_status(input_data)
        if prediction[0] == 0:
            st.write('Prediction: No Default')
        else:
            st.write('Prediction: Default')


if __name__ == '__main__':
    main()

