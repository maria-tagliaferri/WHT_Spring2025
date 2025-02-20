# name: data_processing.py
# description: Process data from .csv files and store them into a single .pkl file``
# author: Vu Phan


import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict
import pickle

import constants
from utils.preprocessing import *


def parse_processing_arguments():
    """ Arguments entered by users to define the data processing step
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', dest = 'data_type', help = 'Type of data, inertial data (imu) or joint kinematics (jk)', type = str, default = 'imu')
    parser.add_argument('--n_sensors', dest = 'n_sensors', help = 'Number of sensors (or IMUs)', type = int, default = 10)
    parser.add_argument('--sensor_pos_1', dest = 'sensor_pos_1', help = 'Sensor position (if n_sensors = 1 or 2)', type = str, default = 'pelvis')
    parser.add_argument('--sensor_pos_2', dest = 'sensor_pos_2', help = 'Sensor position (if n_sensors = 2)', type = str, default = 'thigh_r')
    parser.add_argument('--sensor_mod', dest = 'sensor_mod', help = 'Sensor modalities (acc, gyr, or both)', type = str, default = 'both')
    parser.add_argument('--downsample_factor', dest = 'downsample_factor', help = 'Downsample data', type = int, default = 1)

    return parser.parse_args()


def main_processing(args):
    """ Collect information for data processing
    """ 
    args  = parse_processing_arguments() 

    data_type = args.data_type
    dt_path = get_data_folder(data_type)
    # print folders in dt_path

    if data_type == 'imu':
        downsample_factor = args.downsample_factor # TODO: ADD PROCESSING FOR DATA DOWNSAMPLING
        n_sensors         = args.n_sensors
        sensor_pos_1      = args.sensor_pos_1
        sensor_pos_2      = args.sensor_pos_2
        sensor_mod        = args.sensor_mod

        config      = get_sensor_config(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)
        sample_list = get_sample_list_imu(dt_path, config)
        
        filename = get_data_filename_imu(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)

    elif data_type == 'jk':
        config      = get_jk_config()
        sample_list = get_sample_list_jk(dt_path, config)

        filename = 'data_jk.pkl'
    
    else:
        pass

    processed_path = 'data_processed'
    if os.path.exists(processed_path):
        pass
    else:
        mkfolder(processed_path)
        
    with open(processed_path + '/' + filename, 'wb') as f:
        pickle.dump(sample_list, f)

    print('Number of samples = ' + str(len(sample_list)))
    print('Dimension of the first sample: ' + str(sample_list[0].shape))


if __name__ == '__main__':
    """ Run
    """
    main_processing(sys.argv)

