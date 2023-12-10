import requests
import json
import pandas as pd
from datetime import datetime
import os
from fillDataGaps import fill_crypto_quote_gaps


def fetch_and_update_crypto_quote(api_key, symbol, convert, csv_file_path):
    print("API Key in fetch_and_update_crypto_quote:", api_key)  # Debugging line
    """
    Fetches the latest cryptocurrency quote and updates a CSV file.

    :param api_key: CoinMarketCap API key.
    :param symbol: Cryptocurrency symbol (e.g., 'ETH').
    :param convert: Currency to convert to (e.g., 'USD').
    :param csv_file_path: Path to the CSV file where data will be stored.
    """

    url = 'https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical'
    parameters = {
        'symbol': symbol,
        'convert': convert,
        'count': 1
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key
    }

    try:
        response = requests.get(url, params=parameters, headers=headers)
        if response.status_code != 200:
            print(f"Error: HTTP Status Code {response.status_code}. Response: {response.text}")
            return
        data = json.loads(response.text)
        print(response)

        if 'data' in data and symbol in data['data']:
            for item in data['data'][symbol]:
                for quote in item['quotes']:
                    crypto_data = quote['quote'][convert]
                    crypto_data['timestamp'] = quote['timestamp']
                    update_csv(crypto_data, csv_file_path)
        else:
            print("Error: 'data' key not found in the response or symbol not in data.")
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





# Example usage
fetch_and_update_crypto_quote('5687fd87-dc79-4a6b-86d4-5eb5dcab3669', 'ETH', 'USD', 'buildLiveData/0_Data/rawData.csv')
