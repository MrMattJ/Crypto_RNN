import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def splitTestTrainTroughs(SEQUENCE_LENGTH, TRAIN_SPLIT):
    print("\n\n")
    print("Starting data preparation for troughs...")

    # Load the processed data
    data = pd.read_csv('buildLiveDataPT/Data/deep_historic/processed_data_for_lstm.csv', encoding='utf-8')

    # Separate features and trough labels
    features = data.drop(columns=['DateTime', 'is_trough','is_peak']) 
    label = data['is_trough']

    # Normalize/Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = np.round(features_scaled, 3)


    # Print the list of features used in X_train
    print("List of features used in X_train:", features.columns.tolist())

    # Function to create sequences
    def create_sequences(features, label, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            y.append(label[i])
        return np.array(X), np.array(y)

    # Create sequences
    X, y = create_sequences(features_scaled, label, SEQUENCE_LENGTH)

    # Splitting the data into training and testing sets
    train_size = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Saving the training and testing data as .npy files
    np.save('buildLiveDataPT/Data/training/X_train_troughs.npy', X_train)
    np.save('buildLiveDataPT/Data/training/y_train_troughs.npy', y_train)
    np.save('buildLiveDataPT/Data/testing/X_test_troughs.npy', X_test)
    np.save('buildLiveDataPT/Data/testing/y_test_troughs.npy', y_test)

    print("Data preparation for troughs completed and files saved.")
    print("\n\n")

