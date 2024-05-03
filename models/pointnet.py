import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=5):
        super(TNet, self).__init__()
        self.k = k
        # Reduce the number of channels in convolutional layers
        self.conv1 = nn.Conv1d(k, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)
        self.conv3 = nn.Conv1d(32, 64, 1)
        # Simplify the fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, k*k)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batchsize, -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).repeat(batchsize, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNetSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSeg, self).__init__()
        self.tnet = TNet(k=5)

        # Reduce the number of channels in convolutional layers
        self.conv1 = nn.Conv1d(5, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        # Decoder with reduced complexity
        self.conv4 = nn.Conv1d(80, 32, 1)  # Concatenated feature size: 64 + 16
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(32, 16, 1)
        self.bn5 = nn.BatchNorm1d(16)

        self.conv6 = nn.Conv1d(16, num_classes, 1) # +1 for dummy class

    def forward(self, inp):
        x = inp
        B, N, H = x.shape

        # Encoder
        x = x.transpose(1, 2)
        trans = self.tnet(x)
        x = x.transpose(2, 1)
        x = x @ trans
        x = x.transpose(2, 1)

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))

        # Concatenate global and local features
        x_concat = torch.cat([x3, x1], 1)

        # Decoder
        x = F.relu(self.bn4(self.conv4(x_concat)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return x.transpose(2, 1)
