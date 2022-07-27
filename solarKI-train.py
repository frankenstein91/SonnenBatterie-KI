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
    train_measurements = measurements[:int(len(measurements)*0.8)]
    test_measurements = measurements[int(len(measurements)*0.8):]
    train_statistics = statistics[:int(len(statistics)*0.8)]
    test_statistics = statistics[int(len(statistics)*0.8):]
    # print info
    print("Train measurements:")
    print(train_measurements.info())
    print("Test measurements:")
    print(test_measurements.info())
    print("Train statistics:")
    print(train_statistics.info())
    print("Test statistics:")
    print(test_statistics.info())  
    

if __name__ == '__main__':
    main()