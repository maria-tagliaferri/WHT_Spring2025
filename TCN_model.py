import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        if x.shape[-1] != res.shape[-1]:
            res = res[:, :, :x.shape[-1]]
        return x + res


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    - Consists of multiple TCN blocks with increasing dilation rates.
    - Uses Global Average Pooling to reduce temporal dimension before classification.

    Args:
        num_in (int): Number of input channels (sensor feature dimensions).
        num_out (int): Number of output channels (hidden feature size in TCN blocks).
        kernel_size (int): Size of the convolutional kernel.
        num_classes (int): Number of output classes for classification.
        num_layers (int): Number of stacked TCN blocks (default: 3).
        dilation_rate (int): Base dilation rate, doubled at each layer (default: 2).
    """
    def __init__(self, num_in, num_out, kernel_size, num_classes, num_layers=3, dilation_rate=2):
        super(TCN, self).__init__()
        channels = [num_in] + [num_out] * num_layers + [num_classes]
        self.tcn_layers = nn.Sequential(
            *[TCNBlock(channels[i], channels[i+1], kernel_size, dilation=dilation_rate**i) for i in range(num_layers)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_out, num_classes)

    def forward(self, x):
        x = self.tcn_layers(x)  
        x = self.global_avg_pool(x).squeeze(-1)  
        return self.fc(x)
