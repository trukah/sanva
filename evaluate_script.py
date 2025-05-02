import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import load_and_prepare_data, create_sequences, inverse_transform_predictions
from src.utils import calculate_metrics, plot_predictions

# Configuration
CONFIG = {
    'data_path': '../data/XAU_1h_data.csv',
    'model_path': '../models/lstm_model_xau.h5',
    'scaler_path': '../models/scaler_xau.pkl',
    'sequence_length': 24,
    'train_split': 0.8,
    'validation_split': 0.1,
    'feature_columns': ['Open', 'High', 'Low', 'Close']
}

def evaluate_model():
    """Evaluate the model on the test set and visualize results"""
    # Load data
    print("Loading and preparing data...")
    df, scaler = load_and_prepare_data(CONFIG['data_path'], CONFIG['feature_columns'])
    
    # Create sequences
    X_train, X_val, X_test, y_train, y_val, y_test = create_sequences(
        df, 
        CONFIG['sequence_length'], 
        CONFIG['train_split'],
        CONFIG['validation_split']
    )
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(CONFIG['model_path'])
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae, mse, rmse, mape = calculate_metrics(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Convert scaled predictions back to original scale
    y_test_original = inverse_transform_predictions(
        scaler, y_test, CONFIG['feature_columns'], 'Close'
    )
    y_pred_original = inverse_transform_predictions(
        scaler, y_pred, CONFIG['feature_columns'], 'Close'
    )
    
    # Calculate metrics on original scale
    mae_orig, mse_orig, rmse_orig, mape_orig = calculate_metrics(
        y_test_original, y_pred_original
    )
    print("\nMetrics on original scale:")
    print(f"Mean Absolute Error: {mae_orig:.2f} USD")
    print(f"Root Mean Squared Error: {rmse_orig:.2f} USD")
    print(f"Mean Absolute Percentage Error: {mape_orig:.2f}%")
    
    # Plot predictions vs actual (on test set)
    plot_predictions(
        y_test, y_pred, 
        title='XAU/USD Predictions (Normalized Scale)',
        save_path='../models/test_predictions_normalized.png'
    )
    
    # Plot predictions vs actual (original scale)
    plot_predictions(
        y_test_original, y_pred_original,
        title='XAU/USD Predictions (USD)',
        save_path='../models/test_predictions_usd.png'
    )
    
    # Calculate directional accuracy
    direction_actual = np.diff(y_test_original) > 0
    direction_pred = np.diff(y_pred_original) > 0
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    # Plot directional accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(np.diff(y_test_original), label='Actual Price Change', alpha=0.7)
    plt.plot(np.diff(y_pred_original), label='Predicted Price Change', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Price Direction Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price Change (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../models/directional_accuracy.png')
    
    # Save evaluation results
    results = {
        'Test Loss': test_loss,
        'Test MAE': test_mae,
        'MAE (normalized)': mae,
        'RMSE (normalized)': rmse,
        'MAPE (normalized)': mape,
        'MAE (USD)': mae_orig,
        'RMSE (USD)': rmse_orig,
        'MAPE (%)': mape_orig,
        'Directional Accuracy (%)': directional_accuracy
    }
    
    # Save results to CSV
    pd.DataFrame([results]).to_csv('../models/evaluation_results.csv', index=False)
    print("Evaluation results saved to '../models/evaluation_results.csv'")

if __name__ == "__main__":
    evaluate_model()