import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
dataset_file = 'creditcard.csv'  # Make sure this file is in the same directory as app.py
try:
    # Read the dataset
    data = pd.read_csv(dataset_file)
    
    st.title("Credit Card Fraud Detection")
    
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Prepare data for prediction
    if 'Class' in data.columns:
        X = data.drop(columns=['Class'])  # Exclude target column if present
    else:
        X = data
    
    # Make predictions
    predictions = model.predict(X)
    st.subheader("Fraud Detection Results")
    data['Fraud Prediction'] = predictions
    st.write(data[['Fraud Prediction']])

    # Show fraud summary
    fraud_count = data['Fraud Prediction'].sum()
    total_count = len(data)
    st.write(f"Number of fraudulent transactions: {fraud_count}")
    st.write(f"Total transactions: {total_count}")

except FileNotFoundError:
    st.error(f"Dataset file '{dataset_file}' not found. Please ensure it's in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {e}")
