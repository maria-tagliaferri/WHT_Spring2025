# name: network.py
# description: functions for models


import sys
import os
import torch
from torch.utils.data import Dataset

sys.path.append('/Users/sebastianlevy/Documents/Sebastian/CMU/Spring 2T5/Wearable Health Technologies/Code/IMU_Exercise_Prediction')

import constants
from utils.eval import *
from utils.preprocessing import *


class MyDataset(Dataset):
	""" Dataset handler
	"""

	def __init__(self, list_of_samples, to_size, num_classes):
		self.to_size = to_size
		list_of_samples = [normLength(sample, constants.NORM_SAMPLE_LENGTH).T for sample in list_of_samples]

		self.X = [sample[:constants.ID_EXERCISE_LABEL, :] for sample in list_of_samples]

		if num_classes == 10:
			self.Y = [one_hot_encoding(int(sample[constants.ID_CLUSTER_LABEL, :][0]), num_classes) for sample in list_of_samples]
		else:
			self.Y = [one_hot_encoding(int(sample[constants.ID_EXERCISE_LABEL, :][0]), num_classes) for sample in list_of_samples]

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.X[idx]).float()
		y = self.Y[idx]

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if device == 'cuda':
			x = x.to(device)
			y = torch.from_numpy(y)
			y = y.to(device)

		return x, y


def train_loop(dataloader, model, loss_fn, optimizer, num_classes, scheduler):
	""" Training phase
	"""

	global train_mode
	train_mode = True

	size        = len(dataloader.dataset)
	num_batches = len(dataloader)
	train_loss, correct, sched_factor = 0, 0, 0
	train_losses = []

	cm = np.zeros([num_classes, num_classes]) 
	y_truth = []
	y_pred  = []

	for batch, (X, y) in enumerate(dataloader):
		pred = model(X)
		y    = y.type(torch.FloatTensor)

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if device == 'cuda': y = y.cuda()

		loss = loss_fn(pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 20 == 0:
			loss, current = loss.item(), batch * len(X)
			# print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

		temp_correct, temp_cm, temp_y_truth, temp_y_pred = predict(pred, y, num_classes)
		correct     += temp_correct 
		cm          += temp_cm
		y_truth     += temp_y_truth
		y_pred      += temp_y_pred
		train_loss  += loss_fn(pred, y).item()

	train_loss /= num_batches
	train_losses.append(train_loss)
	correct /= size
	scheduler.step(train_loss)
	# print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

	return correct, cm, y_truth, y_pred


def test_loop(dataloader, model, loss_fn, num_classes):
	""" Testing phase
	"""

	global train_mode
	train_mode = False

	size        = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct, size = 0, 0, 0

	cm = np.zeros([num_classes, num_classes]) 
	y_truth = []
	y_pred = []

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			y = y.type(torch.FloatTensor)
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			if device == 'cuda': y = y.cuda()
			test_loss += loss_fn(pred, y).item()

			temp_correct, temp_cm, temp_y_truth, temp_y_pred = predict(pred, y, num_classes)
			correct += temp_correct
			cm      += temp_cm
			y_truth += temp_y_truth
			y_pred  += temp_y_pred
			size    += y.shape[0]

	test_loss /= num_batches
	correct /= size

	return correct, cm, y_truth, y_pred

