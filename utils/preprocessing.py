# name: preprocessing.py
# description: preprocess data (e.g., read, add labels, etc.)


import sys
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from easydict import EasyDict

sys.path.append('/Users/sebastianlevy/Documents/Sebastian/CMU/Spring 2T5/Wearable Health Technologies/Code/IMU_Exercise_Prediction')

import constants


def get_data_folder(dt_type):
	""" Obtain the folder storing data based on the data type
	"""
	if dt_type == 'imu':
		dt_path = 'data/parsed_h5_csv/'
	elif dt_type =='jk':
		dt_path = 'data/parsed_joint_angles/'
	else:
		pass 

	return dt_path


def mkfolder(pth):
	""" Make a new folder if not exist

	Params:
		pth: folder path | str
	
	Returns:
		No returns but create a new folder (if not created)
	"""
	if not os.path.exists(pth):
		os.mkdir(pth)


def load_df_imu(pth):
	""" Load and re-format data from a given path

	Params:
		pth: path to the (.csv) data file | str

	Returns:
		dtframe: data from the specified file | pd.DataFrame
	"""
	dtframe = pd.read_csv(pth)
	dtframe = dtframe.iloc[:, 3:] 

	names = list(dtframe.columns)
	names = [name.split('.')[0] for name in names]
	names_2 = dtframe.iloc[0, :]
	names_3 = dtframe.iloc[1, :]

	for i in range(len(names)):
		names[i] = names[i] + ' ' + names_2[i] + ' ' + names_3[i]

	dtframe = dtframe.iloc[2:, :]
	dtframe.columns = names

	return dtframe


def load_df_jk(pth):
	""" Loat and re-format joint kinematics data from a given path

	Params:
		pth: path to the (.csv) data file | str

	Returns:
		dtframe: data from the specified file | pd.DataFrame
	"""
	dtframe = pd.read_csv(pth)
	dtframe = dtframe.iloc[:, 1:]

	names   = list(dtframe.columns)
	names   = [name.split('.')[0] for name in names]
	names_2 = dtframe.iloc[0, :]
	names_3 = dtframe.iloc[1, :]

	for i in range(len(names)):
		names[i] = names[i] + ' ' + names_2[i] + ' ' + names_3[i]
	
	dtframe = dtframe.iloc[2:, :]
	dtframe.columns = names 

	return dtframe


def slice_df_imu(dtframe, config):
	""" Slice data (for inertial data)
	"""
	cols     = sorted(dtframe.columns)
	req_cols = [col for col in cols if col.split(' ')[0] in config.sensor_position]
	req_cols = [col for col in req_cols if col.split(' ')[1] in config.sensor_modality]
	dtframe  = dtframe.loc[:, req_cols]

	return dtframe


def slice_df_jk(dtframe, config):
	""" Slice data (for joint kinematics)
	"""
	cols     = sorted(dtframe.columns)
	req_cols = [col for col in cols if col.split(' ')[0] in config.joints]
	req_cols = [col for col in req_cols if col.split(' ')[2] in config.dof]
	dtframe  = dtframe.loc[:, req_cols]

	return dtframe


def one_hot_encoding(label, num_clasess):
	""" Apply one-hot encoding to data label
	"""
	encoding = np.zeros(num_clasess)
	encoding[label] = 1

	return encoding


def one_hot_decoding(code):
	""" Decode a code
	"""

	if code.shape[0] > 0:
		label = np.array([np.where(row == 1) for row in code])
	else:
		label = np.argwhere(code == 1)

	return label


def normLength(arr, maxlength):
	""" Normalize data to have the same sample length for the network input
	"""

	new_arr = np.zeros((maxlength, arr.shape[-1]))
	for i in range(arr.shape[-1]):
		a = arr[:, i]
		k = a.shape[0]
		y = np.interp(np.linspace(0, 1, maxlength), np.linspace(0, 1, k), a)
		new_arr[:, i] = y

	return new_arr


def get_cluster_label(ex_code):
	""" Get label for exercise groups
	"""

	cluster_found = False
	cluster_id = 0

	while not cluster_found:
		if ex_code in constants.CLUSTER[cluster_id]:
			cluster_found = True
		else:
			cluster_id += 1

	return cluster_id


def losocv_split_train_list(all_subject_id, test_subject):  
	""" Split data for the LOSOCV scheme
	"""

	train_list = [m for m in all_subject_id if m != test_subject]

	return train_list


def get_sensor_config(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod):
    """ Obtain the configuration for data pre-processing
    """
    config = EasyDict()

    if n_sensors == 10:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis', 'Chest', 'LeftFoot', 'RightFoot', 'LeftWrist', 'RightWrist']
    elif n_sensors == 7:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis', 'LeftFoot', 'RightFoot']
    elif n_sensors == 5:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis']
    elif n_sensors == 4:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh']
    elif n_sensors == 3:
        config.sensor_position = ['LeftThigh', 'RightThigh', 'Pelvis']
    elif n_sensors == 2:
        config.sensor_position = [constants.POS_INPUT_MAPPING[sensor_pos_1], constants.POS_INPUT_MAPPING[sensor_pos_2]] 
    elif n_sensors == 1:
        config.sensor_position = [constants.POS_INPUT_MAPPING[sensor_pos_1]]
    else:
        print('WARNING: The entered n_sensors is not defined, return configurations of 10 sensors...')
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis', 'Chest', 'LeftFoot', 'RightFoot', 'LeftWrist', 'RightWrist']

    if sensor_mod == 'acc':
        config.sensor_modality = ['Accelerometer']
    elif sensor_mod == 'gyr':
        config.sensor_modality = ['Gyroscope']
    elif sensor_mod == 'both':
        config.sensor_modality = ['Accelerometer', 'Gyroscope']
    else: 
        print('WARNING: The entered sensor_mod is not defined, return configurations using both accelerometer and gyroscope information...')
        config.sensor_modality = ['Accelerometer', 'Gyroscope']

    return config


def get_jk_config():
    """ Obtain configurations for forming joint kinematics dataset

    Params:
        No parameters, just use all joint kinematics
    
    Returns:
        config: configurations | EasyDict of list
    """    
    config = EasyDict() 

    config.joints = ['LeftAnkleAngle', 'LeftHipAngle', 'LeftKneeAngle', 'RightAnkleAngle', 'RightHipAngle', 'RightKneeAngle']
    config.dof    = ['X', 'Y', 'Z']

    return config


def get_sample_list_imu(dt_path, config):
    """ Obtain data from all .csv files storing inertial data

    Params:
        dt_path: directory storing all data | str
        config: configurations (sensor position and modality) for data processing | EasyDict

    Returns:
        sample_list: list of all data samples from .csv files | list of np.array
    """
    sample_list = []

    conv_input_size = len(config.sensor_position) * (len(config.sensor_modality)) * constants.NUM_AX_PER_SENSOR
    dt_ncols        = conv_input_size + 3 # three labels, i.e., exercises, cluster, and subject    

    n_subject     = len(constants.SUBJECT_LIST)
    subject_code  = dict(zip(constants.SUBJECT_LIST, list(range(n_subject))))
    n_exercise    = len(constants.EXERCISE_LIST)
    exercise_code = dict(zip(constants.EXERCISE_LIST, list(range(n_exercise))))

    for subject in tqdm(constants.SUBJECT_LIST):
        print()
        print('Collecting data - subject ', subject)

        for exercise in constants.EXERCISE_LIST:
            try:
                exericse_path  = dt_path + subject + '/' + exercise + '/'
                exercise_files = os.listdir(exericse_path)

                for exercise_fn in exercise_files:
                    sample_path      = exericse_path + exercise_fn
                    df               = load_df_imu(sample_path)
                    df               = slice_df_imu(df, config)
                    df['target']     = exercise_code[exercise]
                    df['cluster']    = get_cluster_label(exercise_code[exercise])
                    df['subject_id'] = subject_code[subject]
                    sample           = np.array(df).astype(float)

                    if sample.shape[1] == dt_ncols:
                        sample_list.append(sample)

            except:
                pass
    
    return sample_list


def get_sample_list_jk(dt_path, config):
	""" Obtain data from all .csv files storing joint kinematics data

	Params:
        dt_path: directory storing all data | str
        config: configurations (sensor position and modality) for data processing | EasyDict

    Returns:
        sample_list: list of all data samples from .csv files | list of np.array
	"""
	sample_list = []
	
	conv_input_size = len(config.joints) * (len(config.dof))
	dt_ncols        = 14 + 3 # three labels, i.e., exercises, cluster, and subject 

	n_subject     = len(constants.SUBJECT_LIST)
	subject_code  = dict(zip(constants.SUBJECT_LIST, list(range(n_subject))))
	n_exercise    = len(constants.EXERCISE_LIST)
	exercise_code = dict(zip(constants.EXERCISE_LIST, list(range(n_exercise))))

	for subject in tqdm(constants.SUBJECT_LIST):
		print()
		print('Collecting data - subject ', subject)

		for exercise in constants.EXERCISE_LIST:
			try:
				exericse_path  = dt_path + subject + '/' + exercise + '/'
				exercise_files = os.listdir(exericse_path)

				for exercise_fn in exercise_files:
					sample_path      = exericse_path + exercise_fn
					df               = load_df_jk(sample_path)
					df               = slice_df_jk(df, config)
					df['target']     = exercise_code[exercise]
					df['cluster']    = get_cluster_label(exercise_code[exercise])
					df['subject_id'] = subject_code[subject]
					sample           = np.array(df).astype(float)

					if sample.shape[1] == dt_ncols:
						sample_list.append(sample)

			except:
				pass

	return sample_list


def get_data_filename_imu(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod):
    """ Get filename for the inertial data based on the configuration

    Params:
        n_sensors: number of sensors | int
        sensor_pos_1: sensor position (if n_sensors = 1 or 2) | str
        sensor_pos_2: sensor position (if n_sensors = 2) | str
        sensor_mod: 'acc', 'gyr', or 'both' | str

    Returns:
        filename: unique filename for the inertial data | str
    """
    if n_sensors > 2:
        base_filename = f'data_imu_{n_sensors}_{sensor_mod}'
    elif n_sensors == 2:
        base_filename = f'data_imu_{n_sensors}_{sensor_pos_1}_{sensor_pos_2}_{sensor_mod}'
    elif n_sensors == 1:
        base_filename = f'data_imu_{n_sensors}_{sensor_pos_1}_{sensor_mod}'
    else:
        return None  # Invalid number of sensors
    
    filename = base_filename + '.pkl'
    counter = 1
    
    while os.path.exists(filename):
        filename = f'{base_filename}_{counter}.pkl'
        counter += 1
    
    return filename





