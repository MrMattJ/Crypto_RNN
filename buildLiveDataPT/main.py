import numpy as np 
import tkinter as tk
root = tk.Tk()
root.withdraw()
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import threading
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
import time
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import talib
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fetchLatestQuote import fetch_and_update_crypto_quote
from fillDataGaps import fill_crypto_quote_gaps

# Data Cleaning Hyperparameters
SMA_PERIOD = 14
EMA_PERIOD = 14
RSI_PERIODS = [7, 14, 21]
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BBANDS_PERIOD = 20
BBANDS_NBDEVUP = 2
BBANDS_NBDEVDN = 2
ROLLING_WINDOW_48 = 48
ROLLING_WINDOW_288 = 288
PEAK_THRESHOLD = 0.005
TROUGH_THRESHOLD = 0.005
FORWARD_BUFFER = 1
BACKWARD_BUFFER = 3
NON_PEAK_REDUCTION_FACTOR = 10  # Adjust this value as needed
X_PREVIOUS_QUOTES = 10

# Data Splitting Hyperparameters
SEQUENCE_LENGTH = 288
TRAINING_DATA_PROPORTION = 0.9
SCALING_PRECISION = 3
TRAIN_SPLIT = .8


# Global constant for sequence length
SEQUENCE_LENGTH = 288

# Global Thresholds
HISTORICAL_PEAK_THRESHOLD = 0.75
HISTORICAL_TROUGH_THRESHOLD = 0.75

def predict_on_testing_set(model, X_test, threshold=HISTORICAL_PEAK_THRESHOLD):
    predictions = model.predict(X_test)
    predicted_indices = [i for i, prediction in enumerate(predictions.flatten()) if prediction > threshold]
    return predicted_indices

def predict_troughs_on_testing_set(model, X_test, threshold=HISTORICAL_TROUGH_THRESHOLD):
    predictions = model.predict(X_test)
    predicted_indices = [i for i, prediction in enumerate(predictions.flatten()) if prediction > threshold]
    return predicted_indices



# Load both models for peaks and troughs
model_peaks = tf.keras.models.load_model('buildLiveDataPT/3_Results/lstm_peak_prediction_model.h5')
model_troughs = tf.keras.models.load_model('buildLiveDataPT/3_Results/lstm_trough_prediction_model.h5')


# Add these lines near the top of your script, after loading your models
X_test_peaks = np.load('buildLiveDataPT/0_Data/testing/X_test_peaks.npy')
predicted_peak_indices = predict_on_testing_set(model_peaks, X_test_peaks)


X_test_troughs = np.load('buildLiveDataPT/0_Data/testing/X_test_troughs.npy')  # Assuming a separate file for troughs
predicted_trough_indices = predict_troughs_on_testing_set(model_troughs, X_test_troughs)


def prepare_latest_data(df):
    print("Preparing latest data...")
    df = df.iloc[-SEQUENCE_LENGTH:].copy()

    if len(df) < SEQUENCE_LENGTH:
        print("Insufficient data points for sequence length")
        return np.zeros((1, SEQUENCE_LENGTH, 24))  # Adjust the number of features (24) as needed

    # Calculate technical indicators
    df['SMA_14'] = talib.SMA(df['Price'], timeperiod=SMA_PERIOD)
    df['EMA_14'] = talib.EMA(df['Price'], timeperiod=EMA_PERIOD)
    df['RSI_7'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[0])
    df['RSI_14'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[1])
    df['RSI_21'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[2])
    macd, signal, _ = talib.MACD(df['Price'], fastperiod=MACD_FAST_PERIOD, slowperiod=MACD_SLOW_PERIOD, signalperiod=MACD_SIGNAL_PERIOD)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Price'], timeperiod=BBANDS_PERIOD, nbdevup=BBANDS_NBDEVUP, nbdevdn=BBANDS_NBDEVDN)
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=252).std() * np.sqrt(252)
    df['rolling_max_48'] = df['Price'].rolling(window=ROLLING_WINDOW_48).max()
    df['rolling_min_48'] = df['Price'].rolling(window=ROLLING_WINDOW_48).min()
    df['past_rolling_max_288'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).max()
    df['past_rolling_min_288'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).min()
    df['equal_past_rolling_max'] = df['Price'].rolling(window=X_PREVIOUS_QUOTES, min_periods=1).apply(lambda x: np.any(x[:-1] == x[-1]) if len(x) > 1 else False, raw=True)
    df['equal_past_rolling_min'] = df['Price'].rolling(window=X_PREVIOUS_QUOTES, min_periods=1).apply(lambda x: np.any(x[:-1] == x[-1]) if len(x) > 1 else False, raw=True)
    df['above_past_rolling_max'] = (df['Price'] > df['past_rolling_max_288']).astype(int)
    df['below_past_rolling_min'] = (df['Price'] < df['past_rolling_min_288']).astype(int)
    df['rolling_max'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).max()
    df['rolling_min'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).min()

    # Handle NaN values by backfilling
    df.bfill(inplace=True)

    # Select the same features used during model training
    features = df[['Price', '24hr_Rolling_Volume_USD', 'SMA_14', 'EMA_14', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'BB_upper', 'BB_middle', 'BB_lower', 'log_return', 'Volatility', 'rolling_max_48', 'rolling_min_48', 'past_rolling_max_288', 'past_rolling_min_288', 'equal_past_rolling_max', 'equal_past_rolling_min', 'above_past_rolling_max', 'below_past_rolling_min', 'rolling_max', 'rolling_min']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled.reshape(1, SEQUENCE_LENGTH, -1)

def predict_peak(model, latest_data):
    print("Predicting peak...")
    prediction = model.predict(latest_data)
    print(prediction[0,0])
    return prediction[0, 0]

def predict_trough(model, latest_data):
    print("Predicting trough...")
    prediction = model.predict(latest_data)
    print(prediction[0,0])
    return prediction[0, 0]

def load_initial_data():
    df = pd.read_csv('buildLiveDataPT/0_Data/deep_historic/ETH_full_5min.csv', on_bad_lines='skip')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df[['DateTime', 'Price', '24hr_Rolling_Volume_USD']]

# You need to create a figure and axes initially before the GUI loop
fig, ax = plt.subplots()

def plot_data(df, is_peak, is_trough, past_peak_indices, past_trough_indices, predicted_peak_indices, predicted_trough_indices, fig, ax):
    if df.empty:
        print("No data available for the selected date range.")
        ax.clear()
        return

    print("Plotting data...")
    ax.clear()  # Clear the existing plot before plotting new data
    fig, ax = plt.subplots()

    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    full_df = df.dropna(subset=['DateTime'])

    one_month_ago = datetime.now() - timedelta(days=30)
    df = df[df['DateTime'] > one_month_ago]

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.plot(df['DateTime'], df['Price'], color='lime', linewidth=0.2)

    ax.set_title('Live Crypto Price Trend', color='white', fontsize=6)
    ax.set_xlabel('DateTime', color='white', fontsize=6)
    ax.set_ylabel('Price', color='white', fontsize=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', colors='white', labelsize=5)
    ax.tick_params(axis='y', colors='white', labelsize=5)
    ax.tick_params(axis='both', which='minor', color='gray')

    latest_quote = df.iloc[-1]
    color = 'red' if is_peak else ('blue' if is_trough else 'lime')
    ax.scatter(latest_quote['DateTime'], latest_quote['Price'], color=color, marker='o', s=100)

    for index in past_peak_indices:
        if index < len(full_df):
            past_peak = full_df.iloc[index]
            ax.scatter(past_peak['DateTime'], past_peak['Price'], color='red', marker='x', s=20)

    for index in past_trough_indices:
        if index < len(full_df):
            past_trough = full_df.iloc[index]
            ax.scatter(past_trough['DateTime'], past_trough['Price'], color='blue', marker='x', s=20)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot predicted peaks from testing set
    for index in predicted_peak_indices:
        # Convert index to datetime - adjust this according to your data
        if index < len(df):
                datetime_value = df.iloc[index]['DateTime']
                ax.scatter(datetime_value, df.iloc[index]['Price'], color='yellow', marker='^', s=5)

    # Plot predicted troughs from testing set
    for index in predicted_trough_indices:
        # Convert index to datetime - adjust this according to your data
        if index < len(df):
                datetime_value = df.iloc[index]['DateTime']
                ax.scatter(datetime_value, df.iloc[index]['Price'], color='blue', marker='v', s=5)

    return fig

"""
def update_plot():
    # Logic to update plot based on the selected dates and live data option
    start_offset = start_date_scale.get()
    end_offset = None if live_data_var.get() == 1 else end_date_scale.get()

    start_date = datetime.now() - timedelta(days=start_offset)
    end_date = datetime.now() - timedelta(days=end_offset) if end_offset is not None else datetime.now()

    df = load_initial_data()  # Your existing function to load data
    df = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]

    fig = plot_data(df, False, False, past_peak_indices, past_trough_indices, predicted_peak_indices, predicted_trough_indices)
    canvas.figure = fig
    canvas.draw()

    

# Bind update_plot to changes in sliders or radio button
start_date_scale.config(command=lambda x: update_plot())
end_date_scale.config(command=lambda x: update_plot())
live_data_button.config(command=update_plot)  

"""

def update_countdown(label, time_left):
    if time_left > 0:
        label.config(text=f"Next update in {time_left} seconds")
        label.after(1000, lambda: update_countdown(label, time_left - 1))
    else:
        label.config(text="Updating...")

def find_gaps_in_data(df, expected_interval=timedelta(minutes=5)):
    gaps = []
    for i in range(1, len(df)):
        time_diff = df.iloc[i]['DateTime'] - df.iloc[i - 1]['DateTime']
        if time_diff > expected_interval:
            gap_start = df.iloc[i - 1]['DateTime']
            gap_end = df.iloc[i]['DateTime']
            gaps.append((gap_start, gap_end))
    return gaps

def save_peak_prediction(index):
    with open('past_peak_predictions.txt', 'a') as file:
        file.write(str(index) + '\n')

def load_past_peak_predictions():
    filename = 'past_peak_predictions.txt'
    if not os.path.exists(filename):
        print(f"{filename} not found. Creating a new file.")
        open(filename, 'a').close()  # Create the file if it doesn't exist
        return []
    try:
        with open(filename, 'r') as file:
            return [int(line.strip()) for line in file]
    except FileNotFoundError:
        print("Error occurred while trying to read the file")
        return []

def save_trough_prediction(index):
    with open('past_trough_predictions.txt', 'a') as file:
        file.write(str(index) + '\n')

def load_past_trough_predictions():
    filename = 'past_trough_predictions.txt'
    if not os.path.exists(filename):
        print(f"{filename} not found. Creating a new file.")
        open(filename, 'a').close()  # Create the file if it doesn't exist
        return []
    try:
        with open(filename, 'r') as file:
            return [int(line.strip()) for line in file]
    except FileNotFoundError:
        print("Error occurred while trying to read the file")
        return []

def send_email(subject, message, recipient_email):
    # [Email configuration and sending logic remain the same]
    print("add email mechanism")

def periodic_update(canvas, countdown_label, api_key, symbol, convert, csv_file_path, sequence_length=SEQUENCE_LENGTH):
    print("Performing periodic update...")

    updated_df = load_initial_data()
    current_date = datetime.now()

    gaps = find_gaps_in_data(updated_df)
    for gap in gaps:
        gap_start, gap_end = gap
        if current_date - gap_start <= timedelta(days=30):
            gap_start_str = gap_start.strftime('%Y-%m-%dT%H:%M:%S')
            gap_end_str = gap_end.strftime('%Y-%m-%dT%H:%M:%S')
            fill_crypto_quote_gaps(api_key, symbol, convert, csv_file_path, gap_start_str, gap_end_str)

    latest_data = prepare_latest_data(updated_df)
    peak_prediction = predict_peak(model_peaks, latest_data)
    trough_prediction = predict_trough(model_troughs, latest_data)

    is_peak = 0.75 < peak_prediction
    is_trough = 0.75 < trough_prediction 

    if is_peak:
        peak_index = len(updated_df) - 1
        past_peak_indices.append(peak_index)
        save_peak_prediction(peak_index)

    if is_trough:
        trough_index = len(updated_df) - 1
        past_trough_indices.append(trough_index)
        save_trough_prediction(trough_index)

    # Pass the predicted indices along with other parameters
    fig, ax = plt.subplots()
    fig = plot_data(df, False, False, past_peak_indices, past_trough_indices, predicted_peak_indices, predicted_trough_indices, fig, ax)
    canvas.figure = fig
    canvas.draw()

    canvas._tkcanvas.after(300000, lambda: periodic_update(canvas, countdown_label, api_key, symbol, convert, csv_file_path))
    update_countdown(countdown_label, 300)



def create_gui(api_key, symbol, convert, csv_file_path):
    print("creating tk window object")
    window = tk.Tk()
    print("setting tk window object title")
    window.title("Crypto Price Trend")

    # Input pane (left empty for now)
    print("making input pane")
    input_pane = tk.Frame(window, width=200)
    input_pane.pack(side=tk.LEFT, fill=tk.Y)

    # Plot pane
    print("making plot pane")
    plot_pane = tk.Frame(window)
    plot_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Initial plot
    print("making initial plot")
    df = load_initial_data()
    fig, ax = plt.subplots()
    fig = plot_data(df, False, False, past_peak_indices, past_trough_indices, predicted_peak_indices, predicted_trough_indices, fig, ax)
    canvas = FigureCanvasTkAgg(fig, master=plot_pane)
    widget = canvas.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)

    # Periodic updates and countdown label
    print("setting up periodic update schedule")
    countdown_label = tk.Label(plot_pane, text="Next update in 300 seconds")
    countdown_label.pack()
    canvas._tkcanvas.after(300000, lambda: periodic_update(canvas, countdown_label, api_key, symbol, convert, csv_file_path))
    update_countdown(countdown_label, 300)

    print("starting main loop")    
    window.mainloop()


    ######
    



if __name__ == "__main__":
    print("Starting application...")
    API_KEY = '5687fd87-dc79-4a6b-86d4-5eb5dcab3669'
    SYMBOL = 'ETH'
    CONVERT = 'USD'
    CSV_FILE_PATH = 'path_to_your_csv_file'

    print("calling load_initial_data")
    df = load_initial_data()
    print("calling load_past_peak_predictions")
    past_peak_indices = load_past_peak_predictions()
    print("calling load_past_trough_predictions")
    past_trough_indices = load_past_trough_predictions()
    print("calling create_gui")
    create_gui(API_KEY, SYMBOL, CONVERT, CSV_FILE_PATH)