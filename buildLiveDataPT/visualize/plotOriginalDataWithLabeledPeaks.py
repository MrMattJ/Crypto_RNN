import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the raw data
raw_df = pd.read_csv('buildLiveDataPT/Data/deep_historic/ETH_full_5min.csv')

# Convert 'DateTime' in raw data to datetime
raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'])

# Load the processed data for peak information
processed_df = pd.read_csv('buildLiveDataPT/Data/deep_historic/processed_data_for_lstm.csv')
processed_df['DateTime'] = pd.to_datetime(processed_df['DateTime'])

# Merge the raw data with processed data to get peak information
merged_df = pd.merge(raw_df, processed_df[['DateTime', 'is_peak','is_trough']], on='DateTime', how='left')

# Plotting
plt.figure(figsize=(15, 7))

# Plot 'Close' trend line from raw data
plt.plot(merged_df['DateTime'], merged_df['Price'], label='Price', alpha=0.7)

# Highlight labeled peaks
peak_indices = merged_df[merged_df['is_peak'] == 1].index
plt.scatter(merged_df['DateTime'][peak_indices], merged_df['Price'][peak_indices], color='red', label='Labeled Peaks')

# Highlight labeled troughs
trough_indices = merged_df[merged_df['is_trough'] == 1].index
plt.scatter(merged_df['DateTime'][trough_indices], merged_df['Price'][trough_indices], color='Blue', label='Labeled Troughs')

# Formatting the x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.title('Raw Data Price with Labeled Peaks Highlighted')
plt.xlabel('DateTime')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout for better fit
plt.show()
