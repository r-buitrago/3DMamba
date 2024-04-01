import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes

class NuScenesLidarDataset(Dataset):
    def __init__(self, path, num_classes, max_points=34816, version='v1.0-mini', lidarseg_path='lidarseg', split="test"):
        self.nusc = NuScenes(version=version, dataroot=path, verbose=True)
        self.lidarseg_path = os.path.join(path, lidarseg_path, version)
        self.max_points = max_points
        self.num_classes = num_classes

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx):
        sample = self.nusc.sample[idx]
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        lidar_points = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)

        # Load corresponding segmentation labels
        lidarseg_filepath = os.path.join(self.lidarseg_path, lidar_token + '_lidarseg.bin')
        if os.path.exists(lidarseg_filepath):
            labels = np.fromfile(lidarseg_filepath, dtype=np.uint8)
        else:
            labels = np.zeros(len(lidar_points), dtype=np.uint8)

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