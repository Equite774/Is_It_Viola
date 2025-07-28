# Mixture Masking augmentation, from SpecAugment++ paper
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import uniform, choice

class MixtureMasking(nn.Module):
    def __init__(self, freq_param, time_param, activate_prob):
        super(MixtureMasking, self).__init__()
        self.freq_param = freq_param
        self.time_param = time_param
        self.activate_prob = activate_prob

    def forward(self, x, labels=None):
        if not self.training or uniform(0, 1) > self.activate_prob or labels is None: # Only apply during training with a certain probability
            return x
        x = x.clone()
        batch_size, _, F, T = x.shape
        for i in range(batch_size):
            # Select second example with the same label
            same_label_indices = self.select_same_label_indices(labels, labels[i])
            j = self.select_random_same_label_index(same_label_indices)
            if j is None or j == i:  # If no other example with the same label, skip
                continue

            # Randomly select time and frequency parameters
            dt = torch.randint(low=0, high=self.time_param + 1, size=(1,)).item()
            df = torch.randint(low=0, high=self.freq_param + 1, size=(1,)).item()
            t0 = torch.randint(0, T - dt + 1, size=(1,)).item()
            f0 = torch.randint(0, F - df + 1, size=(1,)).item()

            # Apply masking
            x[i, :, :, t0:t0 + dt] = 0.5*x[i, :, :, t0:t0 + dt] + 0.5*x[j, :, :, t0:t0 + dt]
            x[i, :, f0:f0 + df, :] = 0.5*x[i, :, f0:f0 + df, :] + 0.5*x[j, :, f0:f0 + df, :]
        return x
    
    def __repr__(self):
        return f"MixtureMasking(freq_param={self.freq_param}, time_param={self.time_param}, activate_prob={self.activate_prob})"

    def select_same_label_indices(self, labels, target_label):
        return [i for i, label in enumerate(labels) if label == target_label]
    
    def select_random_same_label_index(self, same_label_indices):
        if not same_label_indices:
            return None
        return choice(same_label_indices)




