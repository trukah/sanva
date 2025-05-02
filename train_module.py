from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, patience=15, model_path=None):
    """
    Train the model with early stopping and checkpointing.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        model_path (str): Path to save the best model
        
    Returns:
        History: Training history
    """
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        
        # Reduce learning rate when progress stalls
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, 
                          min_lr=1e-6, verbose=1)
    ]
    
    # Add model checkpoint if path is provided
    if model_path:
        callbacks.append(
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
