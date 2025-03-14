"""
Configuration:
    - Specify the number of sensors, sensor positions, and sensor modalities.
    - Optionally enable data augmentation with Gaussian noise by setting augment_data=True.
      The noise parameters (mean, stddev) can also be configured.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from easydict import EasyDict

############################
# CONSTANTS
############################
SUBJECT_LIST = ['SUB01', 'SUB02', 'SUB03', 'SUB04']
EXERCISE_LIST = [
    'BulgSq',
    'CMJDL',
    'Run',
    'SplitJump',
    'SqDL',
    'StepDnL',
    'StepUpL',
    'Walk'
]

# Number of axes per sensor modality (e.g., 3 for accelerometer or gyroscope)
NUM_AX_PER_SENSOR = 3

CLUSTER = [[i] for i in range(len(EXERCISE_LIST))]

# Mapping for sensor positions (used when only 1 or 2 sensors are specified)
POS_INPUT_MAPPING = {
    'pelvis': 'Pelvis',
    'thigh_r': 'RightThigh',
    'thigh_l': 'LeftThigh'
}

INPUT_FOLDER = 'testing/reorient_nosegmented/'
OUTPUT_FOLDER = 'processed_data'

# For IMU data configuration (ignored if data_type is 'jk')
n_sensors    = 10       # Options: 10, 7, 5, 4, 3, 2, or 1
sensor_mod   = 'both'   # Options: 'acc', 'gyr', or 'both'
sensor_pos_1 = 'pelvis' # used if n_sensors == 1 or 2
sensor_pos_2 = 'thigh_r'# used if n_sensors == 2

# Downsample factor (if you wish to add downsampling later; currently not applied)
downsample_factor = 1

# Data Augmentation Configuration
augment_data = False       # Set True to apply Gaussian noise to each sample
noise_mean   = 0.0         # Mean of the Gaussian noise
noise_std    = 0.05        # Stddev of the Gaussian noise


def mkfolder(pth):
    """Create a new folder if it does not exist."""
    if not os.path.exists(pth):
        os.mkdir(pth)

def load_df_imu(pth):
    """Load and re-format inertial data from a CSV file."""
    dtframe = pd.read_csv(pth)
    # Skip the first three columns, then reformat header
    dtframe = dtframe.iloc[:, 3:]
    
    names   = list(dtframe.columns)
    names   = [name.split('.')[0] for name in names]
    names_2 = dtframe.iloc[0, :]
    names_3 = dtframe.iloc[1, :]

    for i in range(len(names)):
        names[i] = names[i] + ' ' + str(names_2[i]) + ' ' + str(names_3[i])

    dtframe = dtframe.iloc[2:, :]
    dtframe.columns = names
    return dtframe

def slice_df_imu(dtframe, config):
    """Slice inertial data based on sensor positions and modalities."""
    cols     = sorted(dtframe.columns)
    req_cols = [col for col in cols if col.split(' ')[0] in config.sensor_position]
    req_cols = [col for col in req_cols if col.split(' ')[1] in config.sensor_modality]
    return dtframe.loc[:, req_cols]

def get_cluster_label(ex_code):
    """Get the cluster label for an exercise code."""
    cluster_found = False
    cluster_id = 0
    while not cluster_found:
        if ex_code in CLUSTER[cluster_id]:
            cluster_found = True
        else:
            cluster_id += 1
    return cluster_id

def get_sensor_config(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod):
    """Obtain configuration for inertial data processing."""
    config = EasyDict()
    if n_sensors == 10:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh',
                                  'Pelvis', 'Chest', 'LeftFoot', 'RightFoot', 'LeftWrist', 'RightWrist']
    elif n_sensors == 7:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis', 'LeftFoot', 'RightFoot']
    elif n_sensors == 5:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis']
    elif n_sensors == 4:
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh']
    elif n_sensors == 3:
        config.sensor_position = ['LeftThigh', 'RightThigh', 'Pelvis']
    elif n_sensors == 2:
        config.sensor_position = [POS_INPUT_MAPPING[sensor_pos_1], POS_INPUT_MAPPING[sensor_pos_2]] 
    elif n_sensors == 1:
        config.sensor_position = [POS_INPUT_MAPPING[sensor_pos_1]]
    else:
        print('WARNING: The entered n_sensors is not defined, using default 10 sensors...')
        config.sensor_position = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh',
                                  'Pelvis', 'Chest', 'LeftFoot', 'RightFoot', 'LeftWrist', 'RightWrist']
    
    if sensor_mod == 'acc':
        config.sensor_modality = ['Accelerometer']
    elif sensor_mod == 'gyr':
        config.sensor_modality = ['Gyroscope']
    elif sensor_mod == 'both':
        config.sensor_modality = ['Accelerometer', 'Gyroscope']
    else:
        print('WARNING: The entered sensor_mod is not defined, using both modalities...')
        config.sensor_modality = ['Accelerometer', 'Gyroscope']
    return config

def add_gaussian_noise(time_series, mean=0.0, stddev=1.0):
    """
    Adds Gaussian noise to a 1D array-like time series.

    Parameters:
    - time_series (array-like): A single column of time series data.
    - mean (float): The average value of the noise. Default is 0.0.
    - stddev (float): Standard deviation of noise. Default is 1.0.

    Returns:
    - noisy_series (np.array): Time series with added noise.
    """
    noise = np.random.normal(mean, stddev, len(time_series))
    noisy_series = time_series + noise
    return noisy_series

############################
# SAMPLE LIST EXTRACTION
############################
def get_sample_list_imu(dt_path, config):
    """
    Obtain data samples from all CSV files storing inertial data.
    Expects a folder structure: dt_path/subject/exercise/*.csv
    """
    sample_list = []
    conv_input_size = len(config.sensor_position) * len(config.sensor_modality) * NUM_AX_PER_SENSOR
    dt_ncols        = conv_input_size + 3  # data columns + 3 label columns

    n_subject     = len(SUBJECT_LIST)
    subject_code  = dict(zip(SUBJECT_LIST, list(range(n_subject))))
    n_exercise    = len(EXERCISE_LIST)
    exercise_code = dict(zip(EXERCISE_LIST, list(range(n_exercise))))

    for subject in tqdm(SUBJECT_LIST, desc="Processing IMU subjects"):
        print(f"\nCollecting data for subject: {subject}")
        for exercise in EXERCISE_LIST:
            exercise_path  = os.path.join(dt_path, subject, exercise)
            if not os.path.exists(exercise_path):
                continue
            try:
                exercise_files = os.listdir(exercise_path)
            except Exception as e:
                print(f"Error listing files in {exercise_path}: {e}")
                continue

            for exercise_fn in exercise_files:
                sample_path = os.path.join(exercise_path, exercise_fn)
                try:
                    df = load_df_imu(sample_path)
                    df = slice_df_imu(df, config)
                    df['target']     = exercise_code[exercise]
                    df['cluster']    = get_cluster_label(exercise_code[exercise])
                    df['subject_id'] = subject_code[subject]
                    sample = np.array(df).astype(float)

                    # Check if sample columns match expectation
                    if sample.shape[1] == dt_ncols:
                        # Optionally augment data (add noise) to the feature columns only
                        if augment_data:
                            for col_idx in range(sample.shape[1] - 3):  # exclude last 3 label columns
                                sample[:, col_idx] = add_gaussian_noise(
                                    sample[:, col_idx],
                                    mean=noise_mean,
                                    stddev=noise_std
                                )
                        sample_list.append(sample)
                except Exception as e:
                    print(f"Error processing {sample_path}: {e}")
                    continue

    return sample_list


############################
# FILENAME GENERATION
############################
def get_data_filename_imu(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod):
    """Generate a unique filename for the inertial data pickle file based on configuration."""
    if n_sensors > 2:
        base_filename = f"data_imu_{n_sensors}_{sensor_mod}"
    elif n_sensors == 2:
        base_filename = f"data_imu_{n_sensors}_{sensor_pos_1}_{sensor_pos_2}_{sensor_mod}"
    elif n_sensors == 1:
        base_filename = f"data_imu_{n_sensors}_{sensor_pos_1}_{sensor_mod}"
    else:
        return None  # Invalid number of sensors
    
    filename = base_filename + ".pkl"
    counter = 1
    # Ensure filename is unique in the output folder
    while os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
        filename = f"{base_filename}_{counter}.pkl"
        counter += 1
    return filename

############################
# MAIN PROCESSING PIPELINE
############################
def main():
    # Use the input folder specified by the user
    dt_path = INPUT_FOLDER

    if not os.path.exists(dt_path):
        print(f"Input folder {dt_path} does not exist. Please update the INPUT_FOLDER variable.")
        sys.exit(1)
    
    # Create the output folder if it does not exist
    mkfolder(OUTPUT_FOLDER)
    
    config = get_sensor_config(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)
    sample_list = get_sample_list_imu(dt_path, config)
    filename = get_data_filename_imu(n_sensors, sensor_pos_1, sensor_pos_2, sensor_mod)
    
    # Save the processed sample list as a pickle file
    output_filepath = os.path.join(OUTPUT_FOLDER, filename)
    with open(output_filepath, "wb") as f:
        pickle.dump(sample_list, f)
    
    print(f"\nNumber of samples: {len(sample_list)}")
    if sample_list:
        print(f"Dimension of the first sample: {sample_list[0].shape}")
    print(f"Data successfully saved to {output_filepath}")

if __name__ == "__main__":
    main()
