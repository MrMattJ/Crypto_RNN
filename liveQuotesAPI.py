import requests
import json

url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'

parameters = {
    'symbol': 'BTC',  # Change to the desired cryptocurrency symbol
    'convert': 'USD',
}

headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': '5687fd87-dc79-4a6b-86d4-5eb5dcab3669',  # Replace with your API key
}

try:
    response = requests.get(url, params=parameters, headers=headers)
    data = json.loads(response.text)

    # Add some spaces for separation
    print("\n" + "=" * 50 + "\n")

    # Check if the 'data' key exists in the response
    if 'data' in data:
        cryptocurrency_data = data['data']['BTC'][0]  # Access the first element in the list
        name = cryptocurrency_data['name']
        symbol = cryptocurrency_data['symbol']
        price = cryptocurrency_data['quote']['USD']['price']
        volume_24h = cryptocurrency_data['quote']['USD']['volume_24h']

        # Print the extracted data
        print("Filtered Response:")
        print("Name:", name)
        print("Symbol:", symbol)
        print("Price:", price)
        print("24h Volume:", volume_24h)
    else:
        print("Error: 'data' key not found in the response.")
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.TooManyRedirects) as e:
    print(e)
