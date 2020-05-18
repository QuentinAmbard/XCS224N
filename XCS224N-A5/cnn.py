#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, embed_size: int = 50, m_word: int = 21, k: int = 5, f: int = None):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels=f, kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        X_conv = self.conv1d(X_reshaped)
        X_conv_out = self.maxpool(F.relu(X_conv))
        return torch.squeeze(X_conv_out, -1)

### END YOUR CODE

