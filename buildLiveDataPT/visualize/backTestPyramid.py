import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def backtest_and_save_to_csv_with_pyramiding():
    # Load models and test data
    model_peaks = load_model('buildLiveDataPT/Results/lstm_peak_prediction_model.h5')
    model_troughs = load_model('buildLiveDataPT/Results/lstm_trough_prediction_model.h5')
    X_test_peaks = np.load('buildLiveDataPT/Data/testing/X_test_peaks.npy')
    X_test_troughs = np.load('buildLiveDataPT/Data/testing/X_test_troughs.npy')

    # Make predictions
    predicted_probabilities_peaks = model_peaks.predict(X_test_peaks)
    predicted_probabilities_troughs = model_troughs.predict(X_test_troughs)
    top_250_peak_indices = np.argsort(predicted_probabilities_peaks.flatten())[-150:]
    top_250_trough_indices = np.argsort(predicted_probabilities_troughs.flatten())[-150:]

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
    buy_sell_cycles = []
    shares_bought_list = []
    purchase_amount = 100  # $100 per purchase
    spending_cap = 1000    # $1000 maximum spending cap
    fee_percentage = 0.0035  # 0.35% fee

    for i in range(start_index_test_data, len(original_data)):
        if i in adjusted_trough_indices and sum([x['Amount'] for x in shares_bought_list]) + purchase_amount <= spending_cap:
            # Buy logic: buy shares worth $100, not exceeding cumulative cost of $1000
            fee = purchase_amount * fee_percentage
            actual_purchase_amount = purchase_amount - fee
            shares_bought = actual_purchase_amount / close_prices[i]
            shares_bought_list.append({'Amount': purchase_amount, 'Shares': shares_bought, 'Price': close_prices[i], 'Date': dates[i]})
            buy_sell_cycles.append({
                'Date': dates[i],
                'Action': 'Buy',
                'Price': close_prices[i],
                'Shares': shares_bought,
                'Amount': purchase_amount,
                'Fee': fee
            })
        elif i in adjusted_peak_indices and shares_bought_list:
            # Sell logic: sell shares from the earliest buy trigger
            earliest_buy = shares_bought_list.pop(0)
            total_sell_price = earliest_buy['Shares'] * close_prices[i]
            fee = total_sell_price * fee_percentage
            profit_loss = total_sell_price - earliest_buy['Amount'] - fee
            buy_sell_cycles.append({
                'Date': dates[i],
                'Action': 'Sell',
                'Price': close_prices[i],
                'Shares': earliest_buy['Shares'],
                'Amount': total_sell_price,
                'Profit/Loss': profit_loss,
                'Fee': fee
            })

    # Create DataFrame for backtesting results
    backtesting_results_df = pd.DataFrame(buy_sell_cycles)

    # Save to CSV
    backtesting_results_df.to_csv('buildLiveDataPT/visualize/backtestingResultsPyramid.csv', index=False)

    return backtesting_results_df

# Run the function and get the backtesting results
backtesting_results_pyramid = backtest_and_save_to_csv_with_pyramiding()
print(backtesting_results_pyramid)
