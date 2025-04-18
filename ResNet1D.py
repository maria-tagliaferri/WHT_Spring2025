# Andres Castrillon
#1D ResNet

import torch
import torch.nn as nn


class ResNet1D(nn.Module):
	#Done referencing [1], [2], and [3].
	#[1] He, Kaiming. "Deep Residula Learning for Image Recognition".  Microsoft Research. arXiv. Dec. 2015.
	#[2] A. Castrillon. "Homework 4". Introduction to Deep Learning (24-788). Carnegie Mellon University. 2025.
	#[3] Phan, Vu et al. "Seven Things To Know about Exercise Classification with Inertial Sensing Wearables." IEEE (2024).
	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(ResNet1D, self).__init__()
		num_out = 64
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride, padding = 'same')
		self.relu1		= nn.ReLU()
		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu2		= nn.ReLU()
		self.conv3 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu3		= nn.ReLU()
		self.conv4 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu4		= nn.ReLU()
		self.conv5 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu5		= nn.ReLU()
		self.conv6 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu6		= nn.ReLU()
		self.conv7 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu7		= nn.ReLU()
		self.conv8 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.relu8		= nn.ReLU()

		#Shortcut convolutions
		self.shortconv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride, padding = 'same')
		self.shortconv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.shortconv3 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		self.shortconv4 		= nn.Conv1d(num_out, num_out, kernel_size, stride, padding = 'same')
		
		self.pooling1	= nn.MaxPool1d((pool_size))
		self.flatten	= nn.Flatten()
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		#Block 1
		
		shortcut1 = x
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		shortcut1 = self.shortconv1(shortcut1)
		x = torch.add(x,shortcut1) #Mimicking how it was done in [2]
		x = self.relu2(x)

		#Block 2
		shortcut2 = x
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.conv4(x)
		shortcut2 = self.shortconv2(shortcut2)
		x = torch.add(x,shortcut2)
		x = self.relu4(x)

		#Block 3
		shortcut3 = x
		x = self.conv5(x)
		x = self.relu5(x)
		x = self.conv6(x)
		shortcut3 = self.shortconv3(shortcut3)
		x = torch.add(x,shortcut3)
		x = self.relu6(x)

		#Block 4
		shortcut4 = x
		x = self.conv7(x)
		x = self.relu7(x)
		x = self.conv8(x)
		shortcut4 = self.shortconv4(shortcut4)
		x = torch.add(x,shortcut4)
		x = self.relu8(x)

		#End
		#Mimicking how it was done in Vu's paper
		x = self.pooling1(x)
		x = self.flatten(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x