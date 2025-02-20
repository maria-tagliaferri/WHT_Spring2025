# name: main.py
# description: Run code here for exercise prediction
# author: Vu Phan


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from torch import nn

from torch.utils.data import DataLoader

from utils.network import *
from utils.eval import *
from utils.preprocessing import *

from model.Type3 import *


def parse_net_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', dest = 'data_type', help = 'Type of data, inertial data (imu) or joint kinematics (jk)', type = str, default = 'imu')
    parser.add_argument('--n_sensors', dest = 'n_sensors', help = 'Number of sensors (or IMUs)', type = int, default = 10)
    parser.add_argument('--sensor_pos_1', dest = 'sensor_pos_1', help = 'Sensor position (if n_sensors = 1 or 2)', type = str, default = 'pelvis')
    parser.add_argument('--sensor_pos_2', dest = 'sensor_pos_2', help = 'Sensor position (if n_sensors = 2)', type = str, default = 'thigh_r')
    parser.add_argument('--sensor_mod', dest = 'sensor_mod', help = 'Sensor modalities (acc, gyr, or both)', type = str, default = 'both')
    parser.add_argument('--downsample_factor', dest = 'downsample_factor', help = 'Downsample data', type = int, default = 1)
    parser.add_argument('--pred_type', dest = 'pred_type', help = 'Predict exercises or groups', type = str, default = 'exercises')

    return parser.parse_args()

def main_net(args):
    args  = parse_net_arguments()

    data_type = args.data_type
    if data_type == 'imu':
        downsample_factor = args.downsample_factor # TODO: ADD PROCESSING FOR DATA DOWNSAMPLING
        n_sensors         = args.n_sensors
        sensor_pos_1      = args.sensor_pos_1
        sensor_pos_2      = args.sensor_pos_2
        sensor_mod        = args.sensor_mod
        config            = get_sensor_config(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)
        data_filename     = get_data_filename_imu(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)

    elif data_type == 'jk':
        config        = get_jk_config()
        data_filename = 'data_jk.pkl'
    
    else:
        pass

    pred_type = args.pred_type
    if pred_type == 'exercises':
        num_classes = 37
    
    else:
        num_classes = 10

    with open('data_processed/' + data_filename, 'rb') as f:
        sample_list = pickle.load(f)

    all_subject_id = list(range(1, 20))
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    perf_train_acc, perf_test_acc       = [], []
    perf_train_cm, perf_test_cm         = [], []
    perf_test_y_pred, perf_test_y_truth = [], []

    for test_subject in all_subject_id[:]:
        print('# Test subject = ' + str(test_subject))
        train_list, test_list = [], []

        for sample in sample_list:
            if sample[0, constants.ID_SUBJECT_LABEL] in [train_subject for train_subject in all_subject_id if train_subject != test_subject]:
                train_list.append(sample)

            elif sample[0, constants.ID_SUBJECT_LABEL] == test_subject:
                test_list.append(sample)
            
            else:
                pass
        
        print('* Training size = ' + str(len(train_list)))
        print('* Test size = ' + str(len(test_list)))

        print('--- Obtain tuned hyperparameters')
        hp_point      = constants.TUNED_HP[test_subject]
        s_batch_size  = hp_point[constants.ID_BATCH_SIZE]
        s_num_out     = hp_point[constants.ID_NUM_OUT]
        s_kernel_size = hp_point[constants.ID_KERNEL_SIZE]
        s_stride      = hp_point[constants.ID_STRIDE]
        s_pool_size   = hp_point[constants.ID_POOL_SIZE]

        print(' + batch size = ' + str(s_batch_size))
        print(' + conv num out = ' + str(s_num_out))
        print(' + kernel size = ' + str(s_kernel_size))
        print(' + stride length = ' + str(s_stride))
        print(' + pool size = ' + str(s_pool_size))

        train_data = MyDataset(train_list, constants.NORM_SAMPLE_LENGTH, num_classes)
        test_data  = MyDataset(test_list, constants.NORM_SAMPLE_LENGTH, num_classes)

        train_dataloader = DataLoader(train_data, batch_size = s_batch_size, shuffle = True)
        test_dataloader  = DataLoader(test_data, batch_size = s_batch_size, shuffle = False)

        conv_num_in = len(config.sensor_position) * (len(config.sensor_modality)) * constants.NUM_AX_PER_SENSOR
        model = CNN_Alter_Block(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes)
        if device == 'cuda': model = model.cuda()

        train_losses, val_losses = [], []

        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = constants.LEARNING_RATE, weight_decay = constants.ADAM_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = constants.LEARNING_RATE_REDUCTION_FACTOR)

        print('--- Start the performance evaluation')
        for t in range(constants.NUM_EPOCHS):
            print(f'Epoch {t + 1}\n---------------------')
            temp_train_acc, temp_train_cm, _, _ = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes, scheduler)
            temp_test_acc, temp_test_cm, temp_y_truth, temp_y_pred = test_loop(test_dataloader, model, loss_fn, num_classes)

            print('*** Training/testing performance')
            perf_train_acc.append(temp_train_acc)
            perf_test_acc.append(temp_test_acc)
            print(perf_train_acc)
            print(perf_test_acc)

            perf_train_cm.append(temp_train_cm)
            perf_test_cm.append(temp_test_cm)
            perf_test_y_truth.append(temp_y_truth)
            perf_test_y_pred.append(temp_y_pred)


if __name__ == '__main__':
    main_net(sys.argv)



