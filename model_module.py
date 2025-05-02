from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def build_lstm_model(input_shape, output_size=1):
    """
    Build an LSTM model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        output_size (int): Number of output units
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(64, 
                   return_sequences=True, 
                   input_shape=input_shape,
                   recurrent_dropout=0.0))  # Using 0.0 for better TFLite compatibility
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(32, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def build_bidirectional_lstm_model(input_shape, output_size=1):
    """
    Build a bidirectional LSTM model for potentially better performance.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        output_size (int): Number of output units
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential()
    
    # First bidirectional LSTM layer
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Second bidirectional LSTM layer
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def build_lightweight_lstm_model(input_shape, output_size=1):
    """
    Build a lightweight LSTM model suitable for TFLite deployment.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, num_features)
        output_size (int): Number of output units
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential()
    
    # Single LSTM layer
    model.add(LSTM(32, input_shape=input_shape))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(output_size))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
