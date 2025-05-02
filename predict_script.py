import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import load_and_prepare_data, inverse_transform_predictions

# Configuration
CONFIG = {
    'data_path': '../data/XAU_1h_data.csv',
    'model_path': '../models/lstm_model_xau.h5',
    'scaler_path': '../models/scaler_xau.pkl',
    'sequence_length': 24,
    'forecast_horizon': 24,  # Hours to forecast ahead
    'feature_columns': ['Open', 'High', 'Low', 'Close']
}

def load_model_and_scaler():
    """Load the trained model and scaler"""
    model = tf.keras.models.load_model(CONFIG['model_path'])
    scaler = joblib.load(CONFIG['scaler_path'])
    return model, scaler

def prepare_last_sequence(df, sequence_length):
    """Extract the last sequence from the dataframe for prediction"""
    # Get the last 'sequence_length' rows
    last_sequence = df.iloc[-sequence_length:].values
    
    # Reshape for the model (adding batch dimension)
    return np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

def recursive_forecast(model, initial_sequence, steps, scaler, feature_columns):
    """
    Make recursive predictions for multiple steps ahead.
    
    Args:
        model: Trained model
        initial_sequence: Starting sequence for prediction
        steps: Number of steps to predict ahead
        scaler: Fitted scaler object
        feature_columns: Feature column names
        
    Returns:
        list: Forecasted values (original scale)
    """
    current_sequence = initial_sequence.copy()
    forecast = []
    
    for _ in range(steps):
        # Predict next time step
        next_pred = model.predict(current_sequence)
        forecast.append(next_pred[0, 0])
        
        # Create a new prediction row (we'll use the last value for all features as placeholder)
        next_row = np.repeat(next_pred[0, 0], len(feature_columns))
        next_row = np.reshape(next_row, (1, 1, len(feature_columns)))
        
        # Update sequence by removing the first time step and adding the prediction
        current_sequence = np.concatenate([current_sequence[:, 1:, :], next_row], axis=1)
    
    # Convert forecasted values back to original scale
    dummy = np.zeros((len(forecast), len(feature_columns)))
    dummy[:, feature_columns.index('Close')] = forecast
    dummy_inverse = scaler.inverse_transform(dummy)
    
    return dummy_inverse[:, feature_columns.index('Close')]

def main():
    # Load data
    print("Loading data...")
    df, scaler = load_and_prepare_data(CONFIG['data_path'], CONFIG['feature_columns'])
    
    # Load model
    print("Loading model...")
    model, _ = load_model_and_scaler()  # We'll use the scaler from data preparation
    
    # Get the last sequence for prediction
    last_sequence = prepare_last_sequence(df, CONFIG['sequence_length'])
    
    # Make recursive forecast
    print(f"Forecasting {CONFIG['forecast_horizon']} hours ahead...")
    forecast = recursive_forecast(
        model, 
        last_sequence, 
        CONFIG['forecast_horizon'], 
        scaler,
        CONFIG['feature_columns']
    )
    
    # Create forecast timestamps (starting from the last data point)
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(hours=i+1) for i in range(len(forecast))]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Time': forecast_dates,
        'Forecasted_Close': forecast
    })
    forecast_df.set_index('Time', inplace=True)
    
    # Print forecast
    print("\nXAU/USD Price Forecast:")
    print(forecast_df)
    
    # Get original prices for last portion of data for comparison
    original_data = pd.read_csv(CONFIG['data_path'])
    if 'Time' in original_data.columns:
        original_data['Time'] = pd.to_datetime(original_data['Time'])
        original_data.set_index('Time', inplace=True)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot recent historical data
    plt.plot(original_data['Close'].iloc[-48:], label='Historical Close Price')
    
    # Plot forecast
    plt.plot(forecast_df['Forecasted_Close'], label='Forecasted Close Price', linestyle='--')
    
    plt.title('XAU/USD Price Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../models/forecast_plot.png')
    print("Forecast plot saved to '../models/forecast_plot.png'")
    
    # Save forecast to CSV
    forecast_df.to_csv('../models/latest_forecast.csv')
    print("Forecast data saved to '../models/latest_forecast.csv'")

if __name__ == "__main__":
    main()
