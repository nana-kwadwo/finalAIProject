import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown 

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download the models and scaler
gdown.cached_download("https://drive.google.com/uc?export=download&id=12UF7LkMU34O2gW4K1mF6RVVZzLtUahul", 'gru_model.joblib')
gdown.cached_download("https://drive.google.com/uc?export=download&id=1iI-lEjkPHMniG6C2IyburH2RulV-m5k5", 'lstm_model.joblib')
gdown.cached_download("https://drive.google.com/uc?export=download&id=1gm-8yaktfPtmSBqvRpGWKFvq5LijAlRq", 'scaler.pkl')

# Load the models and scaler
lstm_model = joblib.load('lstm_model.joblib')
gru_model = joblib.load('gru_model.joblib')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv("CFC_traded_sahres_2019_to_date.csv")
df['Daily Date'] = pd.to_datetime(df['Daily Date'], format='%d/%m/%Y')
df = df.sort_values('Daily Date')

# Ensure the 'Daily Date' column is in datetime format
df['Daily Date'] = pd.to_datetime(df['Daily Date'], format='%d/%m/%Y')

# Verify feature names
print("Dataset columns: ", df.columns)
print("Scaler feature names: ", scaler.feature_names_in_)

# Function to create sequences
def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

# Modified function to make predictions
def make_predictions(start_date, end_date, model):
    # Convert input dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Find the index of the last date before start_date in the dataframe
    last_known_date = df[df['Daily Date'] < start_date]['Daily Date'].max()
    if pd.isnull(last_known_date):
        st.error("Start date is earlier than all dates in the dataset. Please choose a later start date.")
        return None
    
    start_index = df[df['Daily Date'] == last_known_date].index[0]
    
    # Get the previous 30 days of data
    data = df.iloc[max(0, start_index-29):start_index+1]
    
    # Drop 'Daily Date' column and ensure the order of columns matches the scaler's
    data = data.drop(columns=['Daily Date'])
    data = data[scaler.feature_names_in_]

    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Create initial sequence
    X = create_sequences(scaled_data, min(30, len(scaled_data)))
    
    predictions = []
    current_sequence = X[-1]
    
    # Predict for each day from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        # Make prediction
        pred = model.predict(np.array([current_sequence]))
        
        # Append prediction
        predictions.append(pred[0][0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred[0]
        
        current_date += timedelta(days=1)
    
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.concatenate((np.array(predictions).reshape(-1, 1), np.zeros((len(predictions), scaled_data.shape[1] - 1))), axis=1))[:, 0]
    
    return predictions

# Streamlit app
st.title('Stock Price Prediction App')

# Date range input
start_date = st.date_input('Select start date for prediction')
end_date = st.date_input('Select end date for prediction')

if start_date and end_date:
    if start_date > end_date:
        st.error('End date must be after start date')
    else:
        # Make predictions
        lstm_preds = make_predictions(start_date, end_date, lstm_model)
        if lstm_preds is not None:
            gru_preds = make_predictions(start_date, end_date, gru_model)
            
            # Calculate average predictions
            avg_preds = (lstm_preds + gru_preds) / 2
            
            # Create date range for predictions
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                'Date': date_range,
                'Predicted Price': avg_preds
            })
            
            # Display the results
            st.write(pred_df)
            
            # Plotting the graph
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot actual prices if available
            actual_data = df[(df['Daily Date'] >= start_date) & (df['Daily Date'] <= end_date)]
            if not actual_data.empty:
                ax.plot(actual_data['Daily Date'], actual_data['Closing Price - VWAP (GH¢)'], label='Actual Price')
            
            # Plot predicted prices
            ax.plot(pred_df['Date'], pred_df['Predicted Price'], label='Predicted Price', color='r', linestyle='--')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price (GH¢)')
            ax.set_title('Actual vs Predicted Closing Price')
            ax.legend()
            
            st.pyplot(fig)

            # Add a warning for future predictions
            if end_date > df['Daily Date'].max():
                st.warning('Predictions for dates beyond the last known date may be less accurate.')
