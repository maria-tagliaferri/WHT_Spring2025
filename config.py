# name: config.py
# description: contains configuration to run the code


""" Set which sensors to be used for the model
"""
CONSIDERED_IMU_POSITION = ['LeftShank', 'RightShank', 'LeftThigh', 'RightThigh', 'Pelvis', 'Chest', 'LeftFoot', 'RightFoot', 'LeftWrist', 'RightWrist']


""" Set which data type to be used
	Frame: Time | Orientation | Accelerometer | Gyroscope | Magnetometer
	(by including the information in the list, you remove it)
	For example, set NOT_CONSIDERED_INFO = ['Time', 'Orientation', 'Accelerometer', 'Magnetometer'] if you wanted to use gyro. data only
"""
NOT_CONSIDERED_INFO = ['Time', 'Orientation', 'Magnetometer'] # use both accel. and gyro. data


