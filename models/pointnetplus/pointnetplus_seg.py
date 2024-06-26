# import torch.nn as nn
# import torch.nn.functional as F
# from .pointnetplus_utils import PointNetSetAbstraction


# class PointNetPlusPlusSeg(nn.Module):
#     def __init__(self,num_class=32,normal_channel=True):
#         super(PointNetPlusPlusSeg, self).__init__()
#         in_channel = 5 if normal_channel else 3
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(256, num_class)

#     def forward(self, xyz):
#         B, _, _ = xyz.shape
#         print(xyz.shape)
#         xyz = xyz.transpose(1, 2)
#         print(xyz.shape)
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         print(xyz.shape, norm.shape)
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         print(l1_xyz.shape, l1_points.shape)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         print(l2_xyz.shape, l2_points.shape)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         print(l3_xyz.shape, l3_points.shape)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)


#         return x, l3_points



# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss

import torch.nn as nn
import torch.nn.functional as F
from .pointnetplus_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class PointNetPlusPlusSeg(nn.Module):
    def __init__(self, num_class = 32):
        super(PointNetPlusPlusSeg, self).__init__()
        
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 5, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1) # for the dummy class

    def forward(self, xyz):
        # print(xyz.shape)
        # i think this expects the shape to be B, C, N instead of B, N, C
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # return x, l4_points
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

# if __name__ == '__main__':
#     import  torch
#     model = get_model(13)
#     xyz = torch.rand(6, 9, 2048)
#     (model(xyz))