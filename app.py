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

# Prediction app
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
    data_to_scale = df.iloc[start_index-30:start_index].drop(columns=['Daily Date'])
    
    # Verify the data shape and columns
    print("Data to scale columns: ", data_to_scale.columns)
    print("Data to scale shape: ", data_to_scale.shape)
    
    # Check if the feature names match
    if not np.array_equal(data_to_scale.columns, scaler.feature_names_in_):
        st.error("Feature names do not match the scaler's feature names.")
        return None
    
    # Transform the data
    scaled_data = scaler.transform(data_to_scale)
    
    # Create sequences
    sequences = create_sequences(scaled_data, 30)
    
    # Make predictions
    predictions = model.predict(sequences)
    
    return predictions


# Streamlit app
st.title('Cocoa Market Price Predictor')

# Add authors' names
st.markdown("<h6 style='text-align: center; color: gray;'>By Nana Nyarko and Ibrahim Dasuki</h6>", unsafe_allow_html=True)

# Add some space
st.write("")

# Date range input
start_date = st.date_input('Select start date')
end_date = st.date_input('Select close date')

if start_date and end_date:
    if start_date > end_date:
        st.error('Close date must be after start date')
    else:
        # Convert start_date and end_date to pandas Timestamp
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

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
