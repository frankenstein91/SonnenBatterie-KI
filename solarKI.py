#! ./KiPython/bin/python
# -*- coding: utf-8 -*-

import argparse
import math
from operator import index
import tensorflow as tf
import numpy as np
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
    

if __name__ == '__main__':
    main()