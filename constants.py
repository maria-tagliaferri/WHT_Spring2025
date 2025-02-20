# name: constants.py
# description: contains useful constants


""" Indices of exercise, cluster, and subject labels in a data file
	E.g., ──────────── DATA ──────────────── | EXERCISE_LABEL | CLUSTER_LABEL | SUBJECT_LABEL
"""
ID_EXERCISE_LABEL 	= -3
ID_CLUSTER_LABEL 	= -2
ID_SUBJECT_LABEL	= -1


""" Sample length for normalization
"""
NORM_SAMPLE_LENGTH = 100


""" Number of axes for each sensor type (i.e., accel. or gyro.)
"""
NUM_AX_PER_SENSOR = 3 # x-, y-, and z-


""" Number of subjects in the dataset
"""
NUM_SUBJECT = 19


""" Number of exercises in the dataset
"""
NUM_EXERCISE = 37


""" Exercise groups 
	Run clustering.py to obtain this
"""
CLUSTER = [[0, 3, 6, 7, 24, 27, 28, 29, 30, 35],	# cluster 1
			[13, 14, 15, 16, 25], 					# cluster 2
			[12, 17, 20, 31, 32, 33, 34, 36],		# cluster 3
			[22],									# cluster 4
			[18, 19, 23, 26],						# cluster 5
			[21],									# cluster 6
			[10, 11],								# cluster 7
			[8, 9],									# cluster 8
			[4, 5],									# cluster 9
			[1, 2]]									# cluster 10


""" Fixed hyper-parameters of the models
"""
LEARNING_RATE 					= 1e-4
NUM_EPOCHS 						= 30
ADAM_WEIGHT_DECAY 				= 1e-2
LEARNING_RATE_REDUCTION_FACTOR	= 0.5


""" Tuned parameters 
	Run tuning.py to obtain this
"""
ID_BATCH_SIZE  = 0
ID_NUM_OUT     = 1
ID_KERNEL_SIZE = 2
ID_STRIDE      = 3
ID_POOL_SIZE   = 4

# Tuned hyperparameters for using 10 IMUs
# TODO: Add tuned hyperparameters for other configurations
TUNED_HP = [[],
            [32, 256, 4, 1, 2],   # subject 1 left for testing
			[64, 256, 4, 1, 2],   # subject 2 left for testing
			[32, 256, 4, 1, 2],   # subject 3 left for testing
			[32, 256, 4, 1, 2],   # subject 4 left for testing
			[32, 128, 4, 1, 2],   # subject 5 left for testing
			[32, 256, 4, 1, 2],   # subject 6 left for testing
			[32, 256, 4, 1, 2],   # subject 7 left for testing
			[32, 256, 4, 1, 2],   # subject 8 left for testing
			[64, 128, 4, 1, 2],   # subject 9 left for testing
			[32, 128, 4, 1, 2],   # subject 10 left for testing
			[64, 256, 4, 1, 2],   # subject 11 left for testing
			[32, 256, 4, 1, 2],   # subject 12 left for testing
			[32, 128, 4, 1, 2],   # subject 13 left for testing
			[32, 256, 4, 1, 2],   # subject 14 left for testing
			[128, 256, 4, 1, 2],  # subject 15 left for testing
			[64, 128, 4, 1, 2],   # subject 16 left for testing
			[64, 256, 4, 1, 2],   # subject 17 left for testing
			[64, 256, 4, 1, 2],   # subject 18 left for testing
			[32, 128, 4, 1, 2]]   # subject 19 left for testing


""" Information
"""
SUBJECT_LIST = ['SUB01', 'SUB02', 'SUB03', 'SUB04', 'SUB05', 'SUB06', 'SUB07', 'SUB08', 'SUB09',
                'SUB10', 'SUB11', 'SUB12', 'SUB13', 'SUB14', 'SUB15', 'SUB16', 'SUB17', 'SUB18', 'SUB19']

EXERCISE_LIST = ['BulgSq', 'CMJDL', 'CMJSL', 'DeclineSq', 'DropJumpDL', 'DropJumpSL', 'DropLandDL', 'DropLandSL',
                 'FwHop', 'FwHopFast', 'FwJump', 'FwJumpFast', 'HeelRaise', 'LatHop', 'LatHopFast', 
                 'LatJump', 'LatJumpFast', 'Lunge', 'MaxHop', 'MaxJump', 'Pose', 'Run', 'RunCut', 'RunDec',
                 'SpainSq', 'SplitJump', 'SportJump', 'SqDL', 'SqHalfDL', 'SqHalfSL', 'SqSL', 
                 'StepDnH', 'StepDnL', 'StepUpH', 'StepUpL', 'SumoSq', 'Walk']


""" MAPPING OF SENSOR POSITION FOR USER INPUTS
"""
POS_INPUT_MAPPING = {'shank_l': 'LeftShank', 'shank_r': 'RightShank', 
					 'thigh_l': 'LeftThigh', 'thigh_r': 'RightThigh', 
					 'pelvis': 'Pelvis', 'chest': 'Chest', 
					 'foot_l': 'LeftFoot', 'foot_r': 'RightFoot', 
					 'wrist_l': 'LeftWrist', 'wrist_r': 'RightWrist'}



