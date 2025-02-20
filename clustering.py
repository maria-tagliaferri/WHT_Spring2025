# name: clustering.py
# description: Group similar exercises together
# author: Vu Phan


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import constants
from utils.preprocessing import *
from utils.clustering_utils import *


def parse_clustering_arguments():
    """ TBD
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', dest = 'data_type', help = 'Type of data, inertial data (imu) or joint kinematics (jk)', type = str, default = 'imu')

    return parser.parse_args()

def main_clustering(args):
    args  = parse_clustering_arguments()

    data_type = args.data_type
    if data_type == 'imu':
        print('* Using IMU data')
        data_filename = 'testing_processed/data_imu_10_both.pkl'
    elif data_type == 'jk':
        print('* Using joint kinematics')
        data_filename = 'data_processed/data_jk.pkl'
    else:
        pass

    with open(data_filename, 'rb') as f:
        sample_list = pickle.load(f)

    avg_data_frame = prepare_data_4_clustering(sample_list)
    fit_cluster = cluster_data(avg_data_frame, method = 'kmeans', num_clusters = 8)


    fig, ax = plt.subplots()
    ax.plot(fit_cluster.labels_, constants.EXERCISE_LIST, '.')

    plt.show()


if __name__ == '__main__':
    main_clustering(sys.argv)
