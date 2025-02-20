# name: eval.py
# description: perform evaluation of exercise prediction, requiring the full dataset


import numpy as np


def predict(some_tensor, labs, num_classes):
	""" Evaluate prediction
	"""

	some_tensor = some_tensor.cpu().detach().numpy()
	labs        = labs.cpu().detach().numpy()

	cm 		= np.zeros([num_classes, num_classes]) # for storing confusion matrix
	y_truth = []
	y_pred 	= []

	count = 0
	for i in range(some_tensor.shape[0]):
		temp_pred  = np.argmax(some_tensor[i])
		temp_truth = np.argmax(labs[i])

		cm[temp_truth, temp_pred] = cm[temp_truth, temp_pred] + 1

		y_truth.append(temp_truth)
		y_pred.append(temp_pred)

		if temp_pred == temp_truth:
			count = count + 1
		else:
			pass

	return count, cm, y_truth, y_pred


def losocv_split_train_list(all_subject_id, test_subject):
    """ Leave one subject out for testing
    """
    train_list = [m for m in all_subject_id if m != test_subject]

    return train_list