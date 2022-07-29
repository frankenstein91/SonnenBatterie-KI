#! ./KiPython/bin/python
# -*- coding: utf-8 -*-

import argparse
import math
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import time, datetime
import os, sys
# import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, LSTM, Embedding, Dropout, concatenate, Flatten, Activation, Conv1D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import model_from_yaml


dataBoundaries = {
    "measurements": {
        "battery_state_of_charge": {
            "min": 0.0,
            "max": 100.0
            }
        },
    "statistics": {
        }
    }

def main():
    # create parser
    parser = argparse.ArgumentParser(description='SolarKI')
    # add arguments
    # file path to measurements & statistics CSV
    parser.add_argument('--measurements', type=str, default='measurements.csv', help='file path to measurements CSV')
    parser.add_argument('--statistics', type=str, default='statistics.csv', help='file path to statistics CSV')
    # add config group
    ConfigArgs = parser.add_argument_group('Config', 'Configurations')
    # CSV delimiter
    ConfigArgs.add_argument('--delimiter', type=str, default=',', help='CSV delimiter')
    # path to checkpoint directory
    ConfigArgs.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='path to checkpoint directory')
    # splitrate of training data
    ConfigArgs.add_argument('--splitrate', type=float, default=0.8, help='splitrate of training data')
    # parse arguments to local variables
    args = parser.parse_args()
    # create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # read both datasets
    measurements = pd.read_csv(args.measurements, delimiter=args.delimiter,parse_dates=["timestamp"], index_col="timestamp")
    statistics = pd.read_csv(args.statistics, delimiter=args.delimiter,parse_dates=["timestamp"], index_col="timestamp")
    # sort datasets by timestamp
    measurements.sort_values(by=['timestamp'], inplace=True, ascending=True)
    statistics.sort_values(by=['timestamp'], inplace=True, ascending=True)
    # recreate index
    measurements.reset_index(inplace=True)
    statistics.reset_index(inplace=True)
    # print overview of datasets
    print("Measurements:")
    print(measurements.describe().transpose())
    print("Statistics:")
    print(statistics.describe().transpose())
    # print info
    print("Measurements:")
    print(measurements.info())
    print("Statistics:")
    print(statistics.info())
    # split datasets into train and test
    train_measurements = measurements[:int(len(measurements)*args.splitrate)]
    test_measurements = measurements[int(len(measurements)*args.splitrate):]
    train_statistics = statistics[:int(len(statistics)*args.splitrate)]
    test_statistics = statistics[int(len(statistics)*args.splitrate):]
    # print info
    print("Train measurements:")
    print(train_measurements.info())
    print("Test measurements:")
    print(test_measurements.info())
    print("Train statistics:")
    print(train_statistics.info())
    print("Test statistics:")
    print(test_statistics.info())
    # create LSTM model for time series prediction of measurements
    # the model should be able to predict the next 48 hours of measurements
    # list of measurements: production, consumption, battery_charge, battery_discharge, grid_feedin, grid_consumption, battery_state_of_charge, direct_consumption
    measurements_model = Sequential()
    measurements_model.add(LSTM(128, input_shape=(1, 8), return_sequences=True))
    measurements_model.add(LSTM(128, return_sequences=True))
    measurements_model.add(LSTM(64, return_sequences=True))
    measurements_model.add(LSTM(64, return_sequences=True))
    # add output layer
    measurements_model.add(Dense(8))
    # compile model
    measurements_model.compile(loss='mean_squared_error', optimizer='adam')
    # print model summary
    print(measurements_model.summary())
    # create LSTM model for time series prediction of statistics
    # the model should be able to predict the next 48 hours of statistics
    # list of statistics: produced_energy, consumed_energy, battery_charged_energy, battery_discharged_energy, grid_feedin_energy, grid_purchase_energy
    statistics_model = Sequential()
    statistics_model.add(LSTM(128, input_shape=(1, 6), return_sequences=True))
    statistics_model.add(LSTM(128, return_sequences=True))
    statistics_model.add(LSTM(64, return_sequences=True))
    statistics_model.add(LSTM(64, return_sequences=True))
    # add output layer
    statistics_model.add(Dense(6))
    # compile model
    statistics_model.compile(loss='mean_squared_error', optimizer='adam')
    # print model summary
    print(statistics_model.summary())
    # create checkpoint callback
    measurements_model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.checkpoint_dir, "measurements_model_checkpoint.h5"), save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
    statistics_model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.checkpoint_dir, "statistics_model_checkpoint.h5"), save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
    # create early stopping callback
    measurements_model_earlystopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    statistics_model_earlystopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    


if __name__ == '__main__':
    main()