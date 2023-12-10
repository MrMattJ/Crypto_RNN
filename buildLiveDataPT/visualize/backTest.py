import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def backtest_and_save_to_csv():
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
    adjusted_peak_indices = sorted(top_250_peak_indices + start_index_test_data)
    adjusted_trough_indices = sorted(top_250_trough_indices + start_index_test_data)

    # Initialize backtesting variables
    shares_held = 0
    buy_sell_cycles = []
    cumulative_cost = 0
    purchase_amount = 100  # $100 per purchase
    spending_cap = 1000    # $1000 maximum spending cap
    fee_percentage = 0.0035  # 0.35% fee

    # Iterate through each day in the test period
    for i in range(start_index_test_data, len(original_data)):
        if i in adjusted_trough_indices and cumulative_cost + purchase_amount <= spending_cap:
            # Buy logic: buy shares worth $100, not exceeding cumulative cost of $1000
            fee = purchase_amount * fee_percentage
            actual_purchase_amount = purchase_amount - fee
            shares_bought = actual_purchase_amount / close_prices[i]
            shares_held += shares_bought
            cumulative_cost += purchase_amount
            buy_sell_cycles.append({
                'Date': dates[i],
                'Action': 'Buy',
                'Price': close_prices[i],
                'Shares Bought': shares_bought,
                'Cumulative Cost': cumulative_cost,
                'Fee': fee
            })
        elif shares_held > 0 and i in adjusted_peak_indices:
            # Sell logic: sell all shares at the first peak
            total_sell_price = shares_held * close_prices[i]
            fee = total_sell_price * fee_percentage
            profit_loss = total_sell_price - cumulative_cost - fee
            buy_sell_cycles.append({
                'Date': dates[i],
                'Action': 'Sell',
                'Price': close_prices[i],
                'Shares Sold': shares_held,
                'Profit/Loss': profit_loss,
                'Fee': fee
            })
            # Reset for the next buy-sell cycle
            shares_held = 0
            cumulative_cost = 0

    # Create DataFrame for backtesting results
    backtesting_results_df = pd.DataFrame(buy_sell_cycles)

    # Save to CSV
    backtesting_results_df.to_csv('buildLiveDataPT/visualize/backtestingResults.csv', index=False)

    # Calculate total profit
    total_profit = backtesting_results_df['Profit/Loss'].sum() if 'Profit/Loss' in backtesting_results_df.columns else 0

    return backtesting_results_df, total_profit


# Run the function and get the backtesting results

