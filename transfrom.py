import numpy as np
import pandas as pd

# Sample IMU A DataFrame (replace with actual data)
imu_a_data = np.genfromtxt(r"/Users/maria/Documents/WHT/Reorient_keaton/AB02_keat/AB03_jump_1_fb_imu_sim.csv", delimiter=',')
output_file_name = "StepUpH_01_rep1.csv"

# Resample imu_a_data from 200Hz to 100Hz
sampling_rate = 200  # Original sampling rate in Hz
target_rate = 100    # Target sampling rate in Hz
downsample_factor = sampling_rate // target_rate

# Downsample the data by taking every nth row
imu_a_data = imu_a_data[::downsample_factor, :]

imu_a_data = pd.DataFrame({

    'time' : imu_a_data[1:,0],'Pelvis_V_ACCX': imu_a_data[1:,1], 'Pelvis_V_ACCY': imu_a_data[1:,2], 'Pelvis_V_ACCZ': imu_a_data[1:,3],
    'Pelvis_V_GYROX': imu_a_data[1:,4], 'Pelvis_V_GYROY': imu_a_data[1:,5], 'Pelvis_V_GYROZ': imu_a_data[1:,6],

    'LShank_V_ACCX': imu_a_data[1:,19], 'LShank_V_ACCY': imu_a_data[1:,20], 'LShank_V_ACCZ': imu_a_data[1:,21],
    'LShank_V_GYROX': imu_a_data[1:,22], 'LShank_V_GYROY': imu_a_data[1:,23], 'LShank_V_GYROZ': imu_a_data[1:,24],

    'RShank_V_ACCX': imu_a_data[1:,25], 'RShank_V_ACCY': imu_a_data[1:,26], 'RShank_V_ACCZ': imu_a_data[1:,27],
    'RShank_V_GYROX': imu_a_data[1:,28], 'RShank_V_GYROY': imu_a_data[1:,29], 'RShank_V_GYROZ': imu_a_data[1:,30],

    'LFoot_V_ACCX': imu_a_data[1:,31], 'LFoot_V_ACCY': imu_a_data[1:,32], 'LFoot_V_ACCZ': imu_a_data[1:,33],
    'LFoot_V_GYROX': imu_a_data[1:,34], 'LFoot_V_GYROY': imu_a_data[1:,35], 'LFoot_V_GYROZ': imu_a_data[1:,36],

    'RFoot_V_ACCX': imu_a_data[1:,37], 'RFoot_V_ACCY': imu_a_data[1:,38], 'RFoot_V_ACCZ': imu_a_data[1:,39],
    'RFoot_V_GYROX': imu_a_data[1:,40], 'RFoot_V_GYROY': imu_a_data[1:,41], 'RFoot_V_GYROZ': imu_a_data[1:,42]
})

# Define transformation matrices for each body part (adjust as needed)
rotation_matrices = {
    'Pelvis': np.array([[0, -1, 0],  # Reverse mapping: y -> z, -x -> y, -z -> x
                         [0, 0, 1], 
                         [-1, 0, 0]]),

    'LShank': np.array([[0, -1, 0],  # Reverse mapping: z -> x, -x -> y, -y -> z
                         [0, 0, -1], 
                         [1, 0, 0]]),

    'RShank': np.array([[0, -1, 0],  # Reverse mapping: z -> x, -x -> y, -y -> z
                         [0, 0, -1], 
                         [1, 0, 0]]),

    'LFoot': np.array([[1, 0, 0],    # Reverse mapping: x -> x, y -> z, -z -> y
                        [0, 0, -1], 
                        [0, 1, 0]]),

    'RFoot': np.array([[1, 0, 0],    # Reverse mapping: -y -> x, -z -> y, x -> z
                        [0, 0, 1], 
                        [0, 1, 0]])
}

# Initialize transformed data storage
imu_b_data = {}

# Apply transformation for each body segment
for part, R in rotation_matrices.items():
    # Extract acceleration and gyroscope data
    imu_b_data[f'{part}_time'] = imu_a_data['time']  # Add time column for each part
    accel_a = imu_a_data[[f'{part}_V_ACCX', f'{part}_V_ACCY', f'{part}_V_ACCZ']].to_numpy().T
    gyro_a = imu_a_data[[f'{part}_V_GYROX', f'{part}_V_GYROY', f'{part}_V_GYROZ']].to_numpy().T
    
    # Apply transformation
    accel_b = R @ accel_a
    gyro_b = R @ gyro_a

    # Store transformed data
      # Add 4 columns of all 1s before acceleration data
    imu_b_data[f'{part}_col1'] = 0
    imu_b_data[f'{part}_col2'] = 0
    imu_b_data[f'{part}_col3'] = 0
    imu_b_data[f'{part}_col4'] = 0

    imu_b_data[f'{part}_V_ACCX'] = accel_b[0, :]
    imu_b_data[f'{part}_V_ACCY'] = accel_b[1, :]
    imu_b_data[f'{part}_V_ACCZ'] = accel_b[2, :]
    
    imu_b_data[f'{part}_V_GYROX'] = gyro_b[0, :]
    imu_b_data[f'{part}_V_GYROY'] = gyro_b[1, :]
    imu_b_data[f'{part}_V_GYROZ'] = gyro_b[2, :]

        # Add 3 columns of all 1s after gyroscope data
    imu_b_data[f'{part}_col5'] = 0
    imu_b_data[f'{part}_col6'] = 0
    imu_b_data[f'{part}_col7'] = 0
    
# Add the time variable to imu_b_data
imu_b_data['time'] = imu_a_data['time']

# Convert transformed data to DataFrame
imu_b_data = pd.DataFrame(imu_b_data)

imu_b_data.insert(0, '', range(0, len(imu_b_data)))  # Insert row numbers starting at 0 as the first column

# Define the multi-row header structure
header_rows = [
    [
        "", "Pelvis", "Pelvis", "Pelvis", "Pelvis", "Pelvis", "Pelvis",
        "LeftShank", "LeftShank", "LeftShank", "LeftShank", "LeftShank", "LeftShank",
        "RightShank", "RightShank", "RightShank", "RightShank", "RightShank", "RightShank",
        "LeftFoot", "LeftFoot", "LeftFoot", "LeftFoot", "LeftFoot", "LeftFoot",
        "RightFoot", "RightFoot", "RightFoot", "RightFoot", "RightFoot", "RightFoot"
    ],
    [
        "", "Time", "Orientation", "Orientation", "Orientation", "Orientation", "Acceleration", "Acceleration", "Acceleration", "Gyroscope", "Gyroscope", "Gyroscope", "Magnetometer", "Magnetometer", "Magnetometer",
        "Time", "Orientation", "Orientation", "Orientation", "Orientation", "Acceleration", "Acceleration", "Acceleration", "Gyroscope", "Gyroscope", "Gyroscope", "Magnetometer", "Magnetometer", "Magnetometer",
        "Time", "Orientation", "Orientation", "Orientation", "Orientation", "Acceleration", "Acceleration", "Acceleration", "Gyroscope", "Gyroscope", "Gyroscope", "Magnetometer", "Magnetometer", "Magnetometer",
        "Time", "Orientation", "Orientation", "Orientation", "Orientation" ,"Acceleration", "Acceleration", "Acceleration", "Gyroscope", "Gyroscope", "Gyroscope", "Magnetometer", "Magnetometer", "Magnetometer",
        "Time", "Orientation", "Orientation", "Orientation", "Orientation", "Acceleration", "Acceleration", "Acceleration", "Gyroscope", "Gyroscope", "Gyroscope", "Magnetometer", "Magnetometer", "Magnetometer"
    ],
    [
        '', "S", "S", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z","X", "Y", "Z",
        "S", "S", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z","X", "Y", "Z",
        "S", "S", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z","X", "Y", "Z",
        "S", "S", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z","X", "Y", "Z",
        "S", "S", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z","X", "Y", "Z"
    ]
]

# Set columns corresponding to "Orientation" and "Magnetometer" to ones
for col_name in imu_b_data.columns:
    if "Orientation" in col_name or "Magnetometer" in col_name:
        imu_b_data[col_name] = 1
# Save imu_b_data with the multi-row header
output_file_path = r'/Users/maria/Documents/WHT/Reorient_keaton/SUB19/FwJump/FwJump_02_rep1.csv'  # Specify the output file path

import csv  # Import the CSV module for writing headers

with open(output_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write the header rows
    writer.writerows(header_rows)
    # Write the data
    imu_b_data.to_csv(f, index=False, header=False)
for col_name in imu_b_data.columns:
    if "Orientation" in col_name or "Magnetometer" in col_name:
        imu_b_data[col_name] = 1
print(f"Transformed data saved to {output_file_path} with matching header structure.")

