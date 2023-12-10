def clear_file_contents(file_path):
    """
    Clears all data from a specified file.

    :param file_path: Path to the file to be cleared.
    """
    try:
        # Open the file in write mode which truncates the file
        with open(file_path, 'w') as file:
            pass  # File is automatically truncated to zero length
        print(f"Data cleared from {file_path}.")
    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage
clear_file_contents('buildLiveData/0_Data/rawData.csv')
clear_file_contents('buildLiveData/0_Data/processed_data_for_lstm.csv')


