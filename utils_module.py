import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def plot_training_history(history):
    """
    Plot the training and validation loss from model training history.
    
    Args:
        history: Keras training history object
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    plt.close()

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        tuple: (MAE, MSE, RMSE, MAPE)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero in MAPE calculation
    epsilon = 1e-10
    mape = mean_absolute_percentage_error(y_true + epsilon, y_pred + epsilon) * 100
    
    return mae, mse, rmse, mape

def plot_predictions(actual, predicted, title='Model Predictions', save_path=None):
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_forecast(historical, forecast, title='XAU/USD Price Forecast', save_path=None):
    """
    Plot historical data with forecast.
    
    Args:
        historical: Historical data
        forecast: Forecasted values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(range(len(historical)), historical, label='Historical', color='blue')
    
    # Plot forecast starting after historical data
    plt.plot(range(len(historical)-1, len(historical) + len(forecast) - 1), 
             np.append(historical[-1:], forecast), 
             label='Forecast', color='red', linestyle='--')
    
    plt.axvline(x=len(historical)-1, color='green', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
