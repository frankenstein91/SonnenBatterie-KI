#! ./KiPython/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential

dataBoundaries = {
    "measurements": {"battery_state_of_charge": {"min": 0.0, "max": 100.0}},
    "statistics": {},
}


def main():
    # create parser
    parser = argparse.ArgumentParser(description="SolarKI")
    # add arguments
    # file path to measurements & statistics CSV
    parser.add_argument(
        "--measurements",
        type=str,
        default="measurements.csv",
        help="file path to measurements CSV",
    )
    parser.add_argument(
        "--statistics",
        type=str,
        default="statistics.csv",
        help="file path to statistics CSV",
    )
    # add config group
    ConfigArgs = parser.add_argument_group("Config", "Configurations")
    # CSV delimiter
    ConfigArgs.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    # path to checkpoint directory
    ConfigArgs.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint",
        help="path to checkpoint directory",
    )
    # path to the log directory
    ConfigArgs.add_argument(
        "--log_dir", type=str, default="log", help="path to the log directory"
    )
    # path to the model directory
    ConfigArgs.add_argument(
        "--model_dir", type=str, default="model", help="path to the model directory"
    )
    # splitrate of training data
    ConfigArgs.add_argument(
        "--splitrate", type=float, default=0.8, help="splitrate of training data"
    )
    # output_dir
    ConfigArgs.add_argument(
        "--output_dir", type=str, default="output", help="output directory"
    )

    # parse arguments to local variables
    args = parser.parse_args()
    # create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # create log directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # create sub dirs if not exist
    if not os.path.exists(os.path.join(args.log_dir, "measurements_model")):
        os.makedirs(os.path.join(args.log_dir, "measurements_model"))
    if not os.path.exists(os.path.join(args.log_dir, "statistics_model")):
        os.makedirs(os.path.join(args.log_dir, "statistics_model"))
    # create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # read both datasets
    measurements = pd.read_csv(
        args.measurements,
        delimiter=args.delimiter,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    statistics = pd.read_csv(
        args.statistics,
        delimiter=args.delimiter,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    # sort datasets by timestamp
    measurements.sort_values(by=["timestamp"], inplace=True, ascending=True)
    statistics.sort_values(by=["timestamp"], inplace=True, ascending=True)

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
    train_measurements = measurements[: int(len(measurements) * args.splitrate)]
    test_measurements = measurements[int(len(measurements) * args.splitrate) :]
    train_statistics = statistics[: int(len(statistics) * args.splitrate)]
    test_statistics = statistics[int(len(statistics) * args.splitrate) :]
    # print info
    print("Train measurements:")
    print(train_measurements.info())
    print("Test measurements:")
    print(test_measurements.info())
    print("Train statistics:")
    print(train_statistics.info())
    print("Test statistics:")
    print(test_statistics.info())
    # print head of datasets
    print("Train measurements:")
    print(train_measurements.head())
    print("Test measurements:")
    print(test_measurements.head())
    print("Train statistics:")
    print(train_statistics.head())
    print("Test statistics:")
    print(test_statistics.head())
    # convert datasets to tensors
    train_measurements_tensor = tf.convert_to_tensor(
        train_measurements.to_numpy().reshape(-1, 1, len(train_measurements.columns))
    )
    test_measurements_tensor = tf.convert_to_tensor(
        test_measurements.to_numpy().reshape(-1, 1, len(test_measurements.columns))
    )
    train_statistics_tensor = tf.convert_to_tensor(
        train_statistics.to_numpy().reshape(-1, 1, len(train_statistics.columns))
    )
    test_statistics_tensor = tf.convert_to_tensor(
        test_statistics.to_numpy().reshape(-1, 1, len(test_statistics.columns))
    )

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
    measurements_model.compile(loss="mean_squared_error", optimizer="adam")
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
    statistics_model.compile(loss="mean_squared_error", optimizer="adam")
    # print model summary
    print(statistics_model.summary())
    # create checkpoint callback
    measurements_model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "measurements_model_checkpoint.h5"),
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    statistics_model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "statistics_model_checkpoint.h5"),
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    # create early stopping callback
    measurements_model_earlystopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    statistics_model_earlystopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    # create tensorboard callback
    measurements_model_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.log_dir, "measurements_model"), histogram_freq=1
    )
    statistics_model_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.log_dir, "statistics_model"), histogram_freq=1
    )

    # train models
    measurements_model.fit(
        train_measurements_tensor,
        train_measurements_tensor,
        epochs=150,
        batch_size=32,
        validation_data=(test_measurements_tensor, test_measurements_tensor),
        callbacks=[
            measurements_model_checkpoint_callback,
            measurements_model_earlystopping_callback,
            measurements_model_tensorboard_callback,
        ],
    )
    statistics_model.fit(
        train_statistics_tensor,
        train_statistics_tensor,
        epochs=150,
        batch_size=32,
        validation_data=(test_statistics_tensor, test_statistics_tensor),
        callbacks=[
            statistics_model_checkpoint_callback,
            statistics_model_earlystopping_callback,
            statistics_model_tensorboard_callback,
        ],
    )

    # evaluate models
    measurements_model_loss = measurements_model.evaluate(
        test_measurements_tensor, test_measurements_tensor, verbose=0
    )
    statistics_model_loss = statistics_model.evaluate(
        test_statistics_tensor, test_statistics_tensor, verbose=0
    )
    print(f"Measurements model loss: {measurements_model_loss}")
    print(f"Statistics model loss: {statistics_model_loss}")
    # save models
    measurements_model.save(os.path.join(args.model_dir, "measurements_model.h5"))
    statistics_model.save(os.path.join(args.model_dir, "statistics_model.h5"))


if __name__ == "__main__":
    main()
