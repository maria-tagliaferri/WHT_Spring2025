import os
import pandas as pd

def get_exercise_name(folder_name):
    """Convert raw exercise folder names to template naming convention."""
    exercise_map = {
        "Bulgarian": "BulgSq",
        "CounterMovementJump": "CMJDL",
        "Running": "Run",
        "SingleLegSquat": "SqDL",
        "SplitJump": "SplitJump",
        "StepDown": "StepDnL",
        "StepUp": "StepUpL",
        "Walking": "Walk"
    }
    return exercise_map.get(folder_name, folder_name)

# IMU-to-body mapping
imu_mapping = {
    "Chest": "15AE",
    "LeftFoot": "158B",
    "LeftShank": "1584",
    "LeftThigh": "15AB",
    "LeftWrist": "15A9", 
    "Pelvis": "1581",
    "RightFoot": "1590",
    "RightShank": "1587",
    "RightThigh": "15AD",
    "RightWrist": "1586"
}

def find_data_start(file_path):
    """Find the first line containing actual sensor data (skipping metadata)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if not line.startswith("//"):
                return i  # First non-metadata row
    return None  # Should never happen if file is properly formatted

def process_exercise_folder(folder_path, output_folder, sub_id):
    """Process all IMU files in an exercise folder and create a merged CSV file matching the template."""
    imu_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Dictionary to store IMU data
    imu_data = {}

    for imu_name, imu_id in imu_mapping.items():
        matching_file = next((f for f in imu_files if imu_id in f), None)
        
        if matching_file:
            file_path = os.path.join(folder_path, matching_file)
            data_start = find_data_start(file_path)
            if data_start is None:
                print(f"Warning: No valid data found in {file_path}")
                continue

            df = pd.read_csv(file_path, delimiter='\t', skiprows=data_start)

            # Compute Time column using PacketCounter / 120
            df['Time'] = df['PacketCounter'] / 120  # Sampling rate is 120Hz

            # Ensure Time starts at 0 by subtracting the first value
            df['Time'] -= df['Time'].iloc[0]

            # Round Time values to 7 decimal places
            df['Time'] = df['Time'].round(7)

            # Rename columns to match the expected format
            df.rename(columns={
                'Quat_q0': 'Orientation_S',
                'Quat_q1': 'Orientation_X',
                'Quat_q2': 'Orientation_Y',
                'Quat_q3': 'Orientation_Z',
                'Acc_X': 'Accelerometer_X',
                'Acc_Y': 'Accelerometer_Y',
                'Acc_Z': 'Accelerometer_Z',
                'Gyr_X': 'Gyroscope_X',
                'Gyr_Y': 'Gyroscope_Y',
                'Gyr_Z': 'Gyroscope_Z',
                'Mag_X': 'Magnetometer_X',
                'Mag_Y': 'Magnetometer_Y',
                'Mag_Z': 'Magnetometer_Z'
            }, inplace=True)

            df = df[['Time', 'Orientation_S', 'Orientation_X', 'Orientation_Y', 'Orientation_Z',
                     'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
                     'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
                     'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z']]
            
            imu_data[imu_name] = df.reset_index(drop=True)

    if not imu_data:
        print(f"No valid IMU data found in {folder_path}, skipping...")
        return

    # Ensure all IMU DataFrames have the same length
    min_length = min(len(df) for df in imu_data.values())
    for key in imu_data:
        imu_data[key] = imu_data[key].iloc[:min_length].reset_index(drop=True)

    # Create multi-level headers to match the expected structure
    header_1 = [""]  # Sensor names (first column empty for packet counter)
    header_2 = [""]  # Data types (first column empty)
    header_3 = [""]  # Axis labels (first column empty)

    merged_columns = []
    for sensor, df in imu_data.items():
        for col in df.columns:
            merged_columns.append(f"{sensor}_{col}")
            if 'Time' in col:
                header_1.append(sensor)
                header_2.append('Time')
                header_3.append('S')
            elif 'Orientation' in col:
                header_1.append(sensor)
                header_2.append('Orientation')
                header_3.append('S' if 'S' in col else col[-1])
            elif 'Accelerometer' in col:
                header_1.append(sensor)
                header_2.append('Accelerometer')
                header_3.append(col[-1])
            elif 'Gyroscope' in col:
                header_1.append(sensor)
                header_2.append('Gyroscope')
                header_3.append(col[-1])
            elif 'Magnetometer' in col:
                header_1.append(sensor)
                header_2.append('Magnetometer')
                header_3.append(col[-1])

    # Merge all IMU data into a single DataFrame
    merged_df = pd.concat(imu_data.values(), axis=1)
    merged_df.columns = merged_columns

    # Insert a packet counter column
    packet_counter = list(range(1, len(merged_df) + 1))
    merged_df.insert(0, "", packet_counter)  # Empty header name for packet counter

    # Insert multi-level headers
    merged_df.loc[-1] = header_3  # Axis labels (X, Y, Z)
    merged_df.loc[-2] = header_2  # Data types
    merged_df.loc[-3] = header_1  # Sensor names
    merged_df = merged_df.sort_index()

    # Get exercise name from the folder
    exercise_name = get_exercise_name(os.path.basename(folder_path))

    # Determine rep count based on files in folder
    rep_count = len(imu_files) // len(imu_mapping)

    # Define output directory
    sub_folder = os.path.join(output_folder, f"SUB{sub_id}", exercise_name)
    os.makedirs(sub_folder, exist_ok=True)

    for rep in range(rep_count):
        output_filename = f"{exercise_name}_01_rep{rep+1}.csv"
        output_path = os.path.join(sub_folder, output_filename)
        merged_df.to_csv(output_path, index=False, header=False, float_format="%.7f")
        print(f"Saved: {output_path}")

# Example usage
participant_root = "testing"  # Root directory for saving
participant_folders = ["Sebastian_IMU"]  # Participant folders

for i, participant_folder in enumerate(participant_folders, start=1):
    sub_id = str(i).zfill(2)  # Create SUBXX naming (SUB01, SUB02, etc.)
    participant_path = os.path.join(participant_root, participant_folder)
    
    if os.path.isdir(participant_path):
        for exercise_folder in os.listdir(participant_path):
            full_exercise_path = os.path.join(participant_path, exercise_folder)
            if os.path.isdir(full_exercise_path):
                process_exercise_folder(full_exercise_path, participant_root, sub_id)