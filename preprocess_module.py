import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(file_path, feature_columns):
    """
    Load and prepare the dataset for time series forecasting.
    
    Args:
        file_path (str): Path to the CSV data file
        feature_columns (list): List of column names to use as features
        
    Returns:
        tuple: (processed_dataframe, scaler_object)
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if needed
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    
    # Select only the relevant features
    df = df[feature_columns]
    
    # Remove any missing values
    df.dropna(inplace=True)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Create a new DataFrame with the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    return scaled_df, scaler

def create_sequences(df, sequence_length, train_split=0.8, val_split=0.1):
    """
    Create sequences for LSTM training.
    
    Args:
        df (DataFrame): Preprocessed and scaled dataframe
        sequence_length (int): Number of time steps in each sequence
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    data = df.values
    X, y = [], []
    
    # Create sequences
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, df.columns.get_loc('Close')])  # Predict Close price
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    train_size = int(len(X) * train_split)
    val_size = int(len(X) * val_split)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def inverse_transform_predictions(scaler, predictions, feature_columns, target_column='Close'):
    """
    Convert scaled predictions back to original scale.
    
    Args:
        scaler (MinMaxScaler): The fitted scaler object
        predictions (ndarray): The model's predictions (scaled)
        feature_columns (list): List of column names
        target_column (str): The target column name
        
    Returns:
        ndarray: The predictions in original scale
    """
    # Create a dummy array with zeros
    dummy = np.zeros((len(predictions), len(feature_columns)))
    
    # Put the predictions in the right column
    target_idx = feature_columns.index(target_column)
    dummy[:, target_idx] = predictions.flatten()
    
    # Inverse transform
    dummy_inverse = scaler.inverse_transform(dummy)
    
    # Return only the target column
    return dummy_inverse[:, target_idx]
