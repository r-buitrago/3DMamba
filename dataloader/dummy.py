import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.log_utils import get_logger

log = get_logger(__name__)

class DummyDataset(Dataset):
    def __init__(self, shape= (34816,5), length = 128, num_classes = 32, if_test=False):
        self.shape = shape
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(self.shape), torch.randint(0, 32, (self.shape[0],))