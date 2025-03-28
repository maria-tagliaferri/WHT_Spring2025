import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
import torch
import seaborn as sns

from torch import nn
from torch.utils.data import DataLoader

from utils.network import *
from utils.eval import *
from utils.preprocessing import *

from model.Type1 import *
from model.Type2 import *
from model.Type3 import *

def plot_confusion_matrix(cm, num_classes, title='Confusion Matrix', save_path=None):
    """
    Plot and optionally save the confusion matrix
    
    Args:
    cm (numpy.ndarray): Confusion matrix
    num_classes (int): Number of classes
    title (str): Title of the plot
    save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Ensure the confusion matrix is converted to integers
    cm_int = cm.astype(int)
    
    sns.heatmap(cm, annot=cm_int, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def parse_net_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', dest = 'data_type', help = 'Type of data, inertial data (imu) or joint kinematics (jk)', type = str, default = 'imu')
    parser.add_argument('--n_sensors', dest = 'n_sensors', help = 'Number of sensors (or IMUs)', type = int, default = 10)
    parser.add_argument('--sensor_pos_1', dest = 'sensor_pos_1', help = 'Sensor position (if n_sensors = 1 or 2)', type = str, default = 'pelvis')
    parser.add_argument('--sensor_pos_2', dest = 'sensor_pos_2', help = 'Sensor position (if n_sensors = 2)', type = str, default = 'thigh_r')
    parser.add_argument('--sensor_mod', dest = 'sensor_mod', help = 'Sensor modalities (acc, gyr, or both)', type = str, default = 'both')
    parser.add_argument('--downsample_factor', dest = 'downsample_factor', help = 'Downsample data', type = int, default = 1)
    parser.add_argument('--pred_type', dest = 'pred_type', help = 'Predict exercises or groups', type = str, default = 'exercises')
    parser.add_argument('--model_type', type = str, default = 'Type3')
    parser.add_argument('--test_subject', type = int, default=19, help='Subject to use as test subject (1-19)')

    return parser.parse_args()

def main_net(argv=None):
    # Use sys.argv if no arguments are passed
    if argv is None:
        argv = sys.argv

    # Parse arguments
    args = parse_net_arguments()

    data_type = args.data_type
    if data_type == 'imu':
        downsample_factor = args.downsample_factor
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
        raise ValueError("Invalid data type")

    pred_type = args.pred_type
    num_classes = 37 if pred_type == 'exercises' else 10

    with open('data_processed/' + data_filename, 'rb') as f:
        sample_list = pickle.load(f)
    
    # Specific test subject
    test_subject = args.test_subject
    
    # Validate test subject is within range
    if test_subject < 1 or test_subject > 19:
        raise ValueError(f"Test subject must be between 1 and 19, got {test_subject}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    print(f'Training on all subjects except {test_subject}, testing on subject {test_subject}')

    # Split data into train and test lists
    train_list, test_list = [], []
    for sample in sample_list:
        subject_id = sample[0, constants.ID_SUBJECT_LABEL] + 1
        if subject_id == test_subject:
            test_list.append(sample)
        else:
            train_list.append(sample)
    
    print('* Training size = ' + str(len(train_list)))
    print('* Test size = ' + str(len(test_list)))

    # Use hyperparameters for the test subject
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
    
    # Model selection based on model type
    if args.model_type == "Type1_two":
        model = CNN_Two_Blocks(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes)
    elif args.model_type == "Type1_para":
        model = CNN_Parallel_Blocks(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes)
    elif args.model_type == "Type2":
        model = CNN_One_Deep_Block(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes)
    elif args.model_type == "Type3":
        model = CNN_Alter_Block(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    if device == 'cuda': model = model.cuda()

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = constants.LEARNING_RATE, weight_decay = constants.ADAM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = constants.LEARNING_RATE_REDUCTION_FACTOR)

    perf_train_acc, perf_test_acc       = [], []
    perf_train_cm, perf_test_cm         = [], []
    perf_test_y_pred, perf_test_y_truth = [], []

    # Training and testing loop
    for t in range(constants.NUM_EPOCHS):
        temp_train_acc, temp_train_cm, _, _ = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes, scheduler)
        temp_test_acc, temp_test_cm, temp_y_truth, temp_y_pred = test_loop(test_dataloader, model, loss_fn, num_classes)

        perf_train_acc.append(temp_train_acc)
        perf_test_acc.append(temp_test_acc)
        perf_train_cm.append(temp_train_cm)
        perf_test_cm.append(temp_test_cm)
        perf_test_y_truth.append(temp_y_truth)
        perf_test_y_pred.append(temp_y_pred)

    # Final epoch results
    final_test_acc = perf_test_acc[-1]
    final_test_cm = perf_test_cm[-1]
    final_y_truth = perf_test_y_truth[-1]
    final_y_pred = perf_test_y_pred[-1]

    print(f'Final test accuracy for subject {test_subject} is {final_test_acc}')

    # Plot confusion matrix
    plot_confusion_matrix(
        final_test_cm, 
        num_classes, 
        title=f'Confusion Matrix - Subject {test_subject} (Accuracy: {final_test_acc:.2%})',
        save_path=f'confusion_matrix_subject_{test_subject}.png'
    )
    
    return final_test_acc, final_test_cm

if __name__ == '__main__':
    main_net()