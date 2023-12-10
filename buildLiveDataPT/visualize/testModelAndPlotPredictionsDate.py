import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
import pandas as pd

def plotModelsAndPredictions(selected_features):
    # Load models and test data
    model_peaks = load_model('buildLiveDataPT/Results/lstm_peak_prediction_model.h5')
    model_troughs = load_model('buildLiveDataPT/Results/lstm_trough_prediction_model.h5')
    X_test_peaks = np.load('buildLiveDataPT/Data/testing/X_test_peaks.npy')
    X_test_troughs = np.load('buildLiveDataPT/Data/testing/X_test_troughs.npy')

    # Make predictions
    predicted_probabilities_peaks = model_peaks.predict(X_test_peaks)
    predicted_probabilities_troughs = model_troughs.predict(X_test_troughs)
    top_250_peak_indices = np.argsort(predicted_probabilities_peaks.flatten())[-100:]
    top_250_trough_indices = np.argsort(predicted_probabilities_troughs.flatten())[-100:]

    # Load original data
    original_data = pd.read_csv('buildLiveDataPT/Data/deep_historic/processed_data_for_lstm.csv')
    dates = pd.to_datetime(original_data['DateTime'])
    close_prices = original_data['Price'].values

    # Determine the starting index for the test data
    start_index_test_data = len(original_data) - len(X_test_peaks)

    # Adjust indices to align with the dates and prices
    adjusted_peak_indices = top_250_peak_indices + start_index_test_data
    adjusted_trough_indices = top_250_trough_indices + start_index_test_data

    # Align the dates and prices
    aligned_dates = dates[-len(X_test_peaks):]
    close_prices_for_plot = close_prices[-len(X_test_peaks):]

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.plot(aligned_dates, close_prices_for_plot, label='Price', alpha=0.5)

    # Highlight adjusted peaks and troughs
    for idx in adjusted_peak_indices:
        plt.scatter(dates[idx], close_prices[idx], color='green', s=10, label='Predicted Peak' if idx == adjusted_peak_indices[0] else "")
    for idx in adjusted_trough_indices:
        plt.scatter(dates[idx], close_prices[idx], color='blue', s=10, label='Predicted Trough' if idx == adjusted_trough_indices[0] else "")

    plt.title('Test Data Close Prices with Predicted Peaks and Troughs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.gcf().autofmt_xdate()

    # Add the list of included features to the plot
    features_text = ', '.join(selected_features)
    plt.annotate(f'Included Features: {features_text}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)

    # Save the plot as an image
    plt.savefig('buildLiveDataPT/visualize/pastModelPlots/predicted_peaks_troughs.png')
    #plt.show()

# Call the function with selected features
selected_features = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']  # Replace with your selected features
plotModelsAndPredictions(selected_features)
