

# Data Features
SMA_PERIOD = 14
EMA_PERIOD = 14
RSI_PERIODS = [7, 14, 21]
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BBANDS_PERIOD = 20
BBANDS_NBDEVUP = 2
BBANDS_NBDEVDN = 2
ROLLING_WINDOW_48 = 48
ROLLING_WINDOW_288 = 288
PEAK_THRESHOLD = 0.005
TROUGH_THRESHOLD = 0.005
FORWARD_BUFFER = 0
BACKWARD_BUFFER = 0
NON_PEAK_REDUCTION_FACTOR = 10  # Adjust this value as needed
X_PREVIOUS_QUOTES = 10
# model design hyperparameters (Example values)
sequence_length = 288
train_split = 0.8
lstm_units = 20
dense_units = 1
optimizer = 'adam'
learning_rate = 0.001
batch_size = 128
epochs = 1
validation_split = 0.2

import itertools
import random
from preProcessing.cleanAddFeatures import cleanAddFeatures
from preProcessing.splitTestTrainPeaks import splitTestTrainPeaks
from preProcessing.splitTestTrainTroughs import splitTestTrainTroughs
from models.buildPeakModel import buildPeakModel
from models.buildTroughModel import buildTroughModel
from visualize.backTest import backtest_and_save_to_csv
#from visualize.testModelAndPlotPredictionsDate import plotModelsAndPredictions

# List of all possible features
ALL_FEATURES = [
    'SMA',
    'EMA',
    'RSI',
    'MACD',
    'BBANDS',
    '24hr_Rolling_Volume_USD',
    'Volatility',
    'RollingMax', 'RollingMin', 'PastRollingMax', 'PastRollingMin',
    'AbovePastRollingMax', 'BelowPastRollingMin',
    'Above10UnitRollingMax', 'Below10UnitRollingMin',
    'Above50UnitRollingMax', 'Below50UnitRollingMin',
    'Above100UnitRollingMax', 'Below100UnitRollingMin',
    'AbsSlope10Vs50', 'AbsSlope10Vs100', 'AbsSlope10Vs200'
]



# Function to run the entire process with a given set of features
def run_process_with_features(feature_combination):
    # Other hyperparameters here
    cleanAddFeatures(
        SMA_PERIOD, EMA_PERIOD, RSI_PERIODS, MACD_FAST_PERIOD, MACD_SLOW_PERIOD,
        MACD_SIGNAL_PERIOD, BBANDS_PERIOD, BBANDS_NBDEVUP, BBANDS_NBDEVDN,
        ROLLING_WINDOW_48, ROLLING_WINDOW_288, PEAK_THRESHOLD,
        FORWARD_BUFFER, BACKWARD_BUFFER, X_PREVIOUS_QUOTES, feature_combination
    )
    splitTestTrainPeaks(sequence_length, train_split)
    splitTestTrainTroughs(sequence_length, train_split)

    peak_train_loss, peak_train_accuracy, peak_test_loss, peak_test_accuracy = buildPeakModel(
        lstm_units, dense_units, optimizer, learning_rate, batch_size, epochs, validation_split
    )
    trough_train_loss, trough_train_accuracy, trough_test_loss, trough_test_accuracy = buildTroughModel(
        lstm_units, dense_units, optimizer, learning_rate, batch_size, epochs, validation_split
    )

    _, total_profit = backtest_and_save_to_csv()

    return (peak_train_loss, peak_train_accuracy, peak_test_loss, peak_test_accuracy, 
            trough_train_loss, trough_train_accuracy, trough_test_loss, trough_test_accuracy, 
            total_profit)

# Infinite loop
while True:
    # Choose a random number of features between 4 and 8
    num_features = random.randint(4, 8)
    print(num_features)
    # Randomly select the features
    selected_features = random.sample(ALL_FEATURES, num_features)
    
    (peak_train_loss, peak_train_accuracy, peak_test_loss, peak_test_accuracy,
    trough_train_loss, trough_train_accuracy, trough_test_loss, trough_test_accuracy, 
    total_profit) = run_process_with_features(selected_features)
    
    with open('buildLiveDataPT/results/featureCombinationResults.txt', 'a') as file:
        file.write(f"Features: {selected_features} - "
                f"Peak Train Loss: {peak_train_loss}, Peak Train Accuracy: {peak_train_accuracy}, "
                f"Peak Test Loss: {peak_test_loss}, Peak Test Accuracy: {peak_test_accuracy}, "
                f"Trough Train Loss: {trough_train_loss}, Trough Train Accuracy: {trough_train_accuracy}, "
                f"Trough Test Loss: {trough_test_loss}, Trough Test Accuracy: {trough_test_accuracy}, "
                f"Total Profit: {total_profit}\n")

    print(f"Completed iteration with features: {selected_features}")

    # Call plot function with selected features
    #plotModelsAndPredictions(selected_features)
