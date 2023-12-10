import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os

def fill_crypto_quote_gaps(api_key, symbol, convert, csv_file_path, time_start, time_end):
    """
    Fetches historical cryptocurrency quotes for a given time range and updates a CSV file.
    Only fills gaps that start within the last month.

    :param api_key: CoinMarketCap API key.
    :param symbol: Cryptocurrency symbol (e.g., 'ETH').
    :param convert: Currency to convert to (e.g., 'USD').
    :param csv_file_path: Path to the CSV file where data will be stored.
    :param time_start: Start time for fetching data.
    :param time_end: End time for fetching data.
    """

    # Convert time_start and time_end from string to datetime
    try:
        gap_start = pd.to_datetime(time_start)
        gap_end = pd.to_datetime(time_end)
    except ValueError:
        print(f"Error converting time_start or time_end to datetime. Received: {time_start}, {time_end}")
        return

    # Check if gap_start is within the last month
    one_month_ago = datetime.now() - timedelta(days=30)
    if gap_start < one_month_ago:
        print(f"Gap starting at {gap_start} is older than one month. Skipping.")
        return

    print(f"Fetching data for {symbol} from {gap_start} to {gap_end} in {convert}")

    url = 'https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical'
    parameters = {
        'symbol': symbol,
        'convert': convert,
        'time_start': gap_start.isoformat(),
        'time_end': gap_end.isoformat(),
        'interval': '5m'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key
    }

    try:
        response = requests.get(url, params=parameters, headers=headers)
        print(f"API Response: {response.status_code}")
        data = json.loads(response.text)

        if 'data' in data and symbol in data['data']:
            print(f"Processing data for {symbol}")
            for item in data['data'][symbol]:
                for quote in item['quotes']:
                    crypto_data = quote['quote'][convert]
                    crypto_data['timestamp'] = quote['timestamp']
                    update_csv(crypto_data, csv_file_path)
            print("Data processing complete.")
        else:
            print("Error: 'data' key not found in the response or symbol not in data.")
            print(response.text)
    except Exception as e:
        print(f"Error fetching data: {e}")

def update_csv(data, file_path):
    """
    Appends data to a CSV file and reorders it by date. Creates the file if it doesn't exist.

    :param data: A dictionary containing the data to append.
    :param file_path: The path to the CSV file.
    """
    new_data_df = pd.DataFrame([data])
    
    if os.path.isfile(file_path) and os.stat(file_path).st_size != 0:
        existing_data_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df

    # Parse 'timestamp' and handle errors
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')

    # Remove rows with NaT in 'timestamp'
    combined_df = combined_df.dropna(subset=['timestamp'])

    # Sort the DataFrame by the 'timestamp' column
    combined_df.sort_values(by='timestamp', inplace=True)
    
    # Write the sorted DataFrame to the CSV file
    combined_df.to_csv(file_path, mode='w', header=True, index=False)
