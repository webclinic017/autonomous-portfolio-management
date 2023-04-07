import argparse
import pandas as pd
from azureml.core import Run  # Import the Run class
from data_preprocessing import Preprocessing
from LSTM_model import LSTM_model

# Argument parsing for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, dest='data_path', help='Path to the dataset')
args = parser.parse_args()

# Load and preprocess the dataset
preprocessing = Preprocessing(dataset_path=args.data_path)
dataset = preprocessing.load_dataset()
returns = preprocessing.select_most_traded_stocks(dataset)
T = 52  # weeks
data = preprocessing.create_sliding_window(returns, T)

# Define and train the LSTM model
lstm_model = LSTM_model(data)
X_train, y_train, X_test, y_test = lstm_model.prepare_data()
lstm_model.create_model()
lstm_model.train_model(X_train, y_train, X_test, y_test)

# Get the current experiment run context
run = Run.get_context()

# Log the model file as an output of the experiment run
run.upload_file(name='outputs/lstm.regression.h5', path_or_stream='results/lstm_model/lstm.regression.h5')

# # Evaluate the model and save the plots and predictions
# test_predict = lstm_model.evaluate_model(X_test, y_test)

# # Log the saved plots and predictions HDF5 file as outputs of the experiment run
# run.upload_file(name='outputs/lstm_reg.png', path_or_stream='results/lstm_model/lstm_reg.png')
# run.upload_file(name='outputs/predictions.h5', path_or_stream='results/lstm_model/predictions.h5')

# Complete the run
run.complete()