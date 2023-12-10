import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop  # Using legacy optimizers
from sklearn.utils.class_weight import compute_class_weight
from .buildingHyperParameters import (
    LSTM_UNITS, DENSE_UNITS, OPTIMIZER, LEARNING_RATE, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
)

def buildTroughModel(LSTM_UNITS, DENSE_UNITS, OPTIMIZER, LEARNING_RATE, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT):
    print("Starting model training for troughs...")

    # Loading the training data for troughs
    X_train = np.load('buildLiveDataPT/Data/training/X_train_troughs.npy')
    y_train = np.load('buildLiveDataPT/Data/training/y_train_troughs.npy')

    # Checking data types
    print("Data type of X_train:", X_train.dtype)
    print("Data type of y_train:", y_train.dtype)

    # Convert to float if they are boolean
    if X_train.dtype == bool:
        X_train = X_train.astype(np.float32)

    if y_train.dtype == bool:
        y_train = y_train.astype(np.float32)

    # Compute class weights for handling class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=DENSE_UNITS, activation='sigmoid'))

    # Selecting the optimizer
    if OPTIMIZER == 'adam':
        optimizer = Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'sgd':
        optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == 'rmsprop':
        optimizer = RMSprop(learning_rate=LEARNING_RATE)
    else:
        raise ValueError("Invalid optimizer choice")

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, class_weight=class_weights_dict)

    # Save the trained model
    model.save('buildLiveDataPT/Results/lstm_trough_prediction_model.h5')

    # Load the testing data
    X_test = np.load('buildLiveDataPT/data/testing/X_test_troughs.npy')
    y_test = np.load('buildLiveDataPT/data/testing/y_test_troughs.npy')

    # Evaluate the model on the testing set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Extract training loss and accuracy from history
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['accuracy'][-1]

    print("Trough model training completed and model saved.")

    # Return training and testing metrics
    return (training_loss, training_accuracy, test_loss, test_accuracy)

