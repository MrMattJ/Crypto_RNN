import requests
import json
import pandas as pd
from datetime import datetime
import os

def fetch_and_update_historical_crypto_quote(api_key, symbol, convert, csv_file_path):
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
        'count': 8999
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key
    }

    try:
        response = requests.get(url, params=parameters, headers=headers)
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
    Appends data to a CSV file. Creates the file if it doesn't exist.

    :param data: A dictionary containing the data to append.
    :param file_path: The path to the CSV file.
    """
    df = pd.DataFrame([data])

    # Check if the file exists and has content
    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        df.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

# Example usage
fetch_and_update_historical_crypto_quote('5687fd87-dc79-4a6b-86d4-5eb5dcab3669', 'ETH', 'USD', 'buildLiveData/0_Data/rawData.csv')
