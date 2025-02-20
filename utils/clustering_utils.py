# name: clustering_utils.py
# description: clustering functions
# author: Vu Phan
# date: 2023/11/09


import sys
import numpy as np 

from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

sys.path.append('/path/to/opensource')

import constants

from utils.preprocessing import normLength

def normalize_data_4_clustering(sample_list):
    """ Make the length of each rep to 100 samples

    Params:
        sample_list (list of np.array): list of all samples in the dataset
    
    Returns:
        norm_sample_list (list of np.array): list of all normalized samples to form a dataset
    """ 
    norm_sample_list = []

    print('* Normalize data')
    for sample in tqdm(sample_list):
        norm_sample = normLength(sample, constants.NORM_SAMPLE_LENGTH)
        norm_sample_list.append(norm_sample)

    assert len(norm_sample_list) == len(sample_list), 'Number of samples in norm_sample_list and sample_list should be the same'
    assert norm_sample_list[0].shape[0] == constants.NORM_SAMPLE_LENGTH, 'Length of the normalized sample should be 100'

    return norm_sample_list


def prepare_data_4_clustering(sample_list):
    """ Concatenate accelerometry and gyroscope data together, then average across participant/exercise/rep

    Params:
        sample_list (list of np.array): list of all samples in the dataset

    Returns:
        avg_data_frame (np.array): each time series represents an exercise
    """
    data_frame = []
    ex_label = []
    avg_data_frame = []

    num_samples = len(sample_list)
    norm_sample_list = normalize_data_4_clustering(sample_list)

    print('* Concatenate accelerometry and gyroscope data together')
    for i in tqdm(range(num_samples)):
        data_frame.append(norm_sample_list[i][:, 0:constants.ID_EXERCISE_LABEL].flatten('F'))
        ex_label.append(norm_sample_list[i][0, constants.ID_EXERCISE_LABEL])

    data_frame = np.array(data_frame)
    exlabel = np.array(ex_label)

    print('* Average data across participants and rep')
    for ex in tqdm(range(constants.NUM_EXERCISE)):
        temp_all_reps = []

        for i in range(num_samples):
            if ex_label[i] == ex:
                temp_all_reps.append(data_frame[i, :])
            else:
                pass 
        
        temp_all_reps = np.array(temp_all_reps)
        avg_data_frame.append(np.mean(temp_all_reps, axis = 0))

    avg_data_frame = np.array(avg_data_frame)

    assert avg_data_frame.shape[0] == constants.NUM_EXERCISE, 'Each exercise should have only one representative sample'

    return avg_data_frame


def cluster_data(avg_data_frame, method = 'kmeans', num_clusters = 10, seed = 0):
    """ Perform data clustering

    Params:
        avg_data_frame (np.array): each time series represents an exercise
        method (str): kmeans or hierarchical
        num_clusters (int): number of clusters

    Returns:
        fit_cluster: fit cluster of exercise
    """
    if method == 'kmeans':
        fit_cluster = KMeans(n_clusters = num_clusters, random_state = seed).fit(avg_data_frame)
    elif method == 'hierarchical':
        fit_cluster = AgglomerativeClustering(n_clusters = num_clusters).fit(avg_data_frame)
    else:
        pass # tbd

    return fit_cluster


def get_clustering_elbow():
    """ Elbow method for selecting the number of cluster (for k-means)
    
    Params:
        tbd
    
    Returns:
        tbd
    """
    pass
