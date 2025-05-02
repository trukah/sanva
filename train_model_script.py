import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import load_and_prepare_data, create_sequences
from src.model import build_lstm_model
from src.train import train_model
from src.utils import plot_training_history, calculate_metrics

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
CONFIG = {
    'data_path': '../data/XAU_1h_data.csv',
    'model_path': '../models/lstm_model_xau.h5',
    'scaler_path': '../models/scaler_xau.pkl',
    'tflite_path': '../models/model.tflite',
    'sequence_length': 24,  # 24 hours of data
    'train_split': 0.8,
    'validation_split': 0.1,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'target_column': 'Close',
    'feature_columns': ['Open', 'High', 'Low', 'Close']
}

def main():
    # Create directories if they don't exist
    os.makedirs('../models', exist_ok=True)
    
    print("Loading and preparing data...")
    df, scaler = load_and_prepare_data(CONFIG['data_path'], CONFIG['feature_columns'])
    
    # Save the scaler for inference
    joblib.dump(scaler, CONFIG['scaler_path'])
    print(f"Scaler saved to {CONFIG['scaler_path']}")
    
    # Create sequences for LSTM
    X_train, X_val, X_test, y_train, y_val, y_test = create_sequences(
        df, 
        CONFIG['sequence_length'], 
        CONFIG['train_split'],
        CONFIG['validation_split']
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_size=1
    )
    
    # Train model
    print("Training model...")
    history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        model_path=CONFIG['model_path']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    mae, mse, rmse, mape = calculate_metrics(y_test, y_pred)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.4f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('XAU/USD Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('../models/prediction_plot.png')
    
    # Convert to TFLite for deployment
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(CONFIG['tflite_path'], 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {CONFIG['tflite_path']}")
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
