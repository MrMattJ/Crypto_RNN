print("Starting Cleaning of New Data")
import pandas as pd
import numpy as np
import talib

def cleanAddFeatures(
        SMA_PERIOD, EMA_PERIOD, RSI_PERIODS, MACD_FAST_PERIOD, MACD_SLOW_PERIOD,
        MACD_SIGNAL_PERIOD, BBANDS_PERIOD, BBANDS_NBDEVUP, BBANDS_NBDEVDN,
        ROLLING_WINDOW_48, ROLLING_WINDOW_288, PEAK_THRESHOLD,
        FORWARD_BUFFER, BACKWARD_BUFFER, X_PREVIOUS_QUOTES, included_features
    ):
    # Load the data
    df = pd.read_csv('buildLiveDataPT/data/deep_historic/ETH_full_5min.csv', encoding='utf-8')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    # Calculate technical indicators based on included_features
    if 'SMA' in included_features:
        df['SMA'] = talib.SMA(df['Price'], timeperiod=SMA_PERIOD)
    if 'EMA' in included_features:
        df['EMA'] = talib.EMA(df['Price'], timeperiod=EMA_PERIOD)
    if 'RSI' in included_features:
        for period in RSI_PERIODS:
            df[f'RSI_{period}'] = talib.RSI(df['Price'], timeperiod=period)
    if 'MACD' in included_features:
        macd, signal, _ = talib.MACD(df['Price'], fastperiod=MACD_FAST_PERIOD, slowperiod=MACD_SLOW_PERIOD, signalperiod=MACD_SIGNAL_PERIOD)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
    if 'BBANDS' in included_features:
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Price'], timeperiod=BBANDS_PERIOD, nbdevup=BBANDS_NBDEVUP, nbdevdn=BBANDS_NBDEVDN)
    if '24hr_Rolling_Volume_USD' in included_features:
        df['24hr_Rolling_Volume_USD'] = df['24hr_Rolling_Volume_USD']
    #create log_return for volatility calc if volatility is included
    if 'Volatility' in included_features:
        df['log_return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
        df['Volatility'] = df['log_return'].rolling(window=252).std() * np.sqrt(252)
        df.drop(columns=['log_return'], inplace=True)
    if 'RollingMax' in included_features:
        df['rolling_max'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).max()
    if 'RollingMin' in included_features:
        df['rolling_min'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).min()
    if 'PastRollingMax' in included_features:
        df['past_rolling_max'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).max()
    if 'PastRollingMin' in included_features:
        df['past_rolling_min'] = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1).min()
    """
    if 'AbovePastRollingMax' in included_features:
        df['above_past_rolling_max'] = (df['Price'] > past_rolling_max_288).astype(int)
    if 'BelowPastRollingMin' in included_features:
        df['below_past_rolling_min'] = (df['Price'] < past_rolling_min_288).astype(int)
    """
    if 'Above10UnitRollingMax' in included_features:
        df['above_10_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=10).max()).astype(int)
    if 'Below10UnitRollingMin' in included_features:
        df['below_10_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=10).min()).astype(int)
    if 'Above50UnitRollingMax' in included_features:
        df['above_50_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=50).max()).astype(int)
    if 'Below50UnitRollingMin' in included_features:
        df['below_50_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=50).min()).astype(int)
    if 'Above100UnitRollingMax' in included_features:
        df['above_100_unit_rolling_max'] = (df['Price'] > df['Price'].rolling(window=100).max()).astype(int)
    if 'Below100UnitRollingMin' in included_features:
        df['below_100_unit_rolling_min'] = (df['Price'] < df['Price'].rolling(window=100).min()).astype(int)
    for window_size in [50, 100, 200]:
        feature_name = f'AbsSlope10Vs{window_size}'
        if feature_name in included_features:
            df[feature_name] = np.abs(df['Price'].diff(10) / 10) > np.abs(df['Price'].diff(window_size) / window_size)


    # Define a window size for identifying peaks and troughs, and calculate rolling max and min
    rolling_max = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).max()
    rolling_min = df['Price'].rolling(window=ROLLING_WINDOW_288, min_periods=1, center=True).min()
    df['is_peak'] = (df['Price'] == rolling_max.astype(int)).astype(int)
    df['is_trough'] = (df['Price'] == rolling_min.astype(int)).astype(int)

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
    df.to_csv('buildLiveDataPT/data/deep_historic/processed_data_for_lstm.csv', index=False)

    print(f"Finished Cleaning. Total points: {len(df)}, Number of peaks: {df['is_peak'].sum()}, Number of troughs: {df['is_trough'].sum()}")

