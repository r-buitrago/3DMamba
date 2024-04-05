import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

from utils.log_utils import get_logger

log = get_logger(__name__)

class NuScenesLidarDataset(Dataset):
    def __init__(self, path, num_classes, max_points=34816, train_version='v1.0-mini', lidarseg_path='lidarseg',
                  test_version='v1.0-mini', if_test=False, max_samples=-1, test_split = None):
        version = train_version if not if_test else test_version
        self.nusc = NuScenes(version=version, dataroot=path, verbose=True)
        self.lidarseg_path = os.path.join(path, lidarseg_path, version)
        self.max_points = max_points
        self.num_classes = num_classes

        # Filtering samples that have corresponding LIDAR data and segmentation labels
        self.available_samples = []
        for sample in self.nusc.sample:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_filepath, boxes_lidar, _ = self.nusc.get_sample_data(lidar_token)
            # lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            lidarseg_filepath = os.path.join(self.lidarseg_path, lidar_token + '_lidarseg.bin')
            if os.path.exists(lidar_filepath) and os.path.exists(lidarseg_filepath):
                self.available_samples.append(sample)
            if max_samples > 0 and len(self.available_samples) >= max_samples:
                break
        
        if train_version == test_version and test_split is not None:
            split = int(len(self.available_samples) * test_split)
            if if_test:
                self.available_samples = self.available_samples[:split]
            else:
                self.available_samples = self.available_samples[split:]

    def __len__(self):
        return len(self.available_samples)

    def __getitem__(self, idx):
        sample = self.available_samples[idx]
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_filepath, boxes_lidar, _ = self.nusc.get_sample_data(lidar_token)
        # lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        lidar_points = np.fromfile(lidar_filepath, dtype=np.float32, count=-1)
        if len(lidar_points) % 5 != 0:
            log.warning(f"Invalid LIDAR data: {lidar_filepath}, shape: {lidar_points.shape}, chopping the last {len(lidar_points) % 5} points")
            # reshape the data to have 5 columns
            lidar_points = lidar_points[:-(len(lidar_points) % 5)]
        
        lidar_points = lidar_points.reshape(-1, 5)
        

        # Load corresponding segmentation labels
        lidarseg_filepath = os.path.join(self.lidarseg_path, lidar_token + '_lidarseg.bin')
        labels = np.fromfile(lidarseg_filepath, dtype=np.uint8)

        # Pad or truncate the point clouds and labels
        num_points = len(lidar_points)
        if num_points < self.max_points:
            pad_size = self.max_points - num_points
            lidar_points = np.pad(lidar_points, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=self.num_classes)  # Padding labels with a null value
        elif num_points > self.max_points:
            lidar_points = lidar_points[:self.max_points]
            labels = labels[:self.max_points]

        # Convert to PyTorch tensors
        lidar_points_tensor = torch.from_numpy(lidar_points).float()
        labels_tensor = torch.from_numpy(labels).long()

        return lidar_points_tensor, labels_tensor
