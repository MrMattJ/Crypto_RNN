import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained models for peaks and troughs
model_peaks = load_model('buildLiveDataPT/Results/lstm_peak_prediction_model.h5')
model_troughs = load_model('buildLiveDataPT/Results/lstm_trough_prediction_model.h5')

# Load the test data for peaks and troughs
X_test_peaks = np.load('buildLiveDataPT/Data/testing/X_test_peaks.npy')
X_test_troughs = np.load('buildLiveDataPT/Data/testing/X_test_troughs.npy')

# Make predictions for peaks and troughs
predicted_probabilities_peaks = model_peaks.predict(X_test_peaks)
predicted_probabilities_troughs = model_troughs.predict(X_test_troughs)

# Select the top 100 peaks and troughs based on their probabilities
top_100_peak_indices = np.argsort(predicted_probabilities_peaks.flatten())[-100:]
top_100_trough_indices = np.argsort(predicted_probabilities_troughs.flatten())[-100:]

# Load the original data to get the 'Close' prices
original_data = pd.read_csv('buildLiveDataPT/Data/deep_historic/processed_data_for_lstm.csv')
close_prices = original_data['Price'].values

# Align the 'Close' prices with the test data
close_prices_for_plot_peaks = close_prices[-len(X_test_peaks):]
close_prices_for_plot_troughs = close_prices[-len(X_test_troughs):]

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(close_prices_for_plot_peaks, label='Price', alpha=0.5)

# Highlight top 100 predicted peaks in green
plt.scatter(top_100_peak_indices, close_prices_for_plot_peaks[top_100_peak_indices], color='green', label='Top 100 Predicted Peaks')

# Highlight top 100 predicted troughs in blue
plt.scatter(top_100_trough_indices, close_prices_for_plot_troughs[top_100_trough_indices], color='blue', label='Top 100 Predicted Troughs')

plt.title('Test Data Close Prices with Top 100 Predicted Peaks and Troughs Highlighted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()