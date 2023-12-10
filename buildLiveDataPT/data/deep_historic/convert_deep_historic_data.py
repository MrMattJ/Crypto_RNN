import pandas as pd

# Path to the text file and output CSV file
input_file_path = 'buildLiveData/0_Data/deep_historic/ETH_full_5min.txt'
output_file_path = 'buildLiveData/0_Data/deep_historic/ETH_full_5min.csv'

# Load the text file
df = pd.read_csv(input_file_path, header=None)

# Add column names
df.columns = ['DateTime', 'Open', 'High', 'Low', 'Price', 'Volume']

# Keep only DateTime, Close, and Volume
df = df[['DateTime', 'Price', 'Volume']]

# Convert DateTime to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Sort the DataFrame by DateTime
df = df.sort_values('DateTime')

# Calculate rolling 24-hour volume in crypto units
# (24 hours * 12 intervals per hour for 5-minute intervals)
rolling_window = 24 * 12
df['24hr_Rolling_Volume_Crypto'] = df['Volume'].rolling(window=rolling_window).sum()

# Convert volume to USD (you'll need to add the conversion logic here)
# Assuming 'conversion_rate' is the rate to convert from crypto units to USD
conversion_rate = 1  # Replace with actual conversion rate
df['24hr_Rolling_Volume_USD'] = df['24hr_Rolling_Volume_Crypto'] * conversion_rate

# Remove the original Volume column
df.drop(columns=['Volume', '24hr_Rolling_Volume_Crypto'], inplace=True)

# Save to CSV
df.to_csv(output_file_path, index=False)

print("File processed and saved as CSV.")
