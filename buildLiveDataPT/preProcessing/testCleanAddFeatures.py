print("Starting Cleaning of New Data")
import pandas as pd
import numpy as np
import talib
from cleaningHyperParameters import (
    SMA_PERIOD, EMA_PERIOD, RSI_PERIODS, MACD_FAST_PERIOD, MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD, BBANDS_PERIOD, BBANDS_NBDEVUP, BBANDS_NBDEVDN,
    ROLLING_WINDOW_48, ROLLING_WINDOW_288, PEAK_THRESHOLD,
    FORWARD_BUFFER, BACKWARD_BUFFER, X_PREVIOUS_QUOTES
)

def cleanAddFeatures():
    # Load the data
    df = pd.read_csv('buildLiveData/Data/deep_historic/ETH_full_5min.csv', encoding='utf-8')

    # Convert 'DateTime' and sort
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    # Calculate technical indicators
    # df['SMA_14'] = talib.SMA(df['Price'], timeperiod=SMA_PERIOD)
    # df['EMA_14'] = talib.EMA(df['Price'], timeperiod=EMA_PERIOD)
    # df['RSI_7'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[0])
    # df['RSI_14'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[1])
    # df['RSI_21'] = talib.RSI(df['Price'], timeperiod=RSI_PERIODS[2])
    # macd, signal, _ = talib.MACD(df['Price'], fastperiod=MACD_FAST_PERIOD, slowperiod=MACD_SLOW_PERIOD, signalperiod=MACD_SIGNAL_PERIOD)
    # df['MACD'] = macd
    # df['MACD_Signal'] = signal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Price'], timeperiod=BBANDS_PERIOD, nbdevup=BBANDS_NBDEVUP, nbdevdn=BBANDS_NBDEVDN)

    # Additional features
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=252).std() * np.sqrt(252)
    # df['rolling_max_48'] = df['Price'].rolling(window=ROLLING_WINDOW_48).max()
    # df['rolling_min_48'] = df['Price'].rolling(window=ROLLING_WINDOW_48).min()
    past_rolling_max_288 = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).max()
    past_rolling_min_288 = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).min()

    # Check if any of the X previous quotes equal the rolling max or min
    df['equal_past_rolling_max'] = df['Price'].rolling(window=X_PREVIOUS_QUOTES, min_periods=1).apply(
        lambda x: np.any(x[:-1] == x[-1]) if len(x) > 1 else False, raw=True
    )
    df['equal_past_rolling_min'] = df['Price'].rolling(window=X_PREVIOUS_QUOTES, min_periods=1).apply(
        lambda x: np.any(x[:-1] == x[-1]) if len(x) > 1 else False, raw=True
    )

    # Binary classification: is the latest quote above the rolling max or below the rolling min
    df['above_past_rolling_max'] = (df['Price'] > past_rolling_max_288)
    df['below_past_rolling_min'] = (df['Price'] < past_rolling_min_288)

    # Add features for the latest price relative to rolling max/min for various window sizes
    df['above_10_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=10).max()).astype(int)
    df['below_10_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=10).min()).astype(int)
    df['above_50_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=50).max()).astype(int)
    df['below_50_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=50).min()).astype(int)
    df['above_100_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=100).max()).astype(int)
    df['below_100_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=100).min()).astype(int)
    # Add features for whether the absolute value of the slope between the latest point and different historical points is greater or less than the absolute value of the slope between the latest point and 10 points ago
    for window_size in [50, 100, 200]:
        df[f'abs_slope_10_vs_{window_size}'] = np.abs(df['Price'].diff(10) / 10) > np.abs(df['Price'].diff(window_size) / window_size)
    # Calculate the slope between the latest point and 10 points ago
    df['slope_10_points'] = np.diff(df['Price'], n=10) / 10


    # Define a window size for identifying peaks and troughs, and calculate rolling max and min
    df['rolling_max'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).max()
    df['rolling_min'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).min()
    df['is_peak'] = (df['Price'] == df['rolling_max']).astype(int)
    df['is_trough'] = (df['Price'] == df['rolling_min']).astype(int)

    # Filter and adjust peaks
    for i in range(len(df)):
        if df['is_peak'].iloc[i] == 1:
            min_before = df['Price'].iloc[max(0, i - ROLLING_WINDOW_288):i].min()
            min_after = df['Price'].iloc[i + 1:min(len(df), i + ROLLING_WINDOW_288 + 1)].min()
            if df['Price'].iloc[i] <= (1 + PEAK_THRESHOLD) * min(min_before, min_after):
                df.at[i, 'is_peak'] = 0
            else:
                for j in range(max(0, i - BACKWARD_BUFFER), min(i + FORWARD_BUFFER, len(df))):
                    df.at[j, 'is_peak'] = 1

    # Filter and adjust troughs
    for i in range(len(df)):
        if df['is_trough'].iloc[i] == 1:
            max_before = df['Price'].iloc[max(0, i - ROLLING_WINDOW_288):i].max()
            max_after = df['Price'].iloc[i + 1:min(len(df), i + ROLLING_WINDOW_288 + 1)].max()
            if df['Price'].iloc[i] >= (1 - PEAK_THRESHOLD) * max(max_before, max_after):
                df.at[i, 'is_trough'] = 0
            else:
                for j in range(max(0, i - BACKWARD_BUFFER), min(i + FORWARD_BUFFER, len(df))):
                    df.at[j, 'is_trough'] = 1

    # Round all numeric columns to 3 decimal places
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].round(3)

    # Keep necessary columns and drop NaNs
    df = df.dropna()

    # Save the cleaned and processed data
    df.to_csv('buildLiveDataPT/Data/deep_historic/processed_data_for_lstm.csv', index=False)

    print(f"Finished Cleaning. Total points: {len(df)}, Number of peaks: {df['is_peak'].sum()}, Number of troughs: {df['is_trough'].sum()}")

