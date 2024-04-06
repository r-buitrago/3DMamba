import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        ''' input: (B, num_classes)
        target: (B, num_classes) (one-hot)
        '''
        B, C = input.shape
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = -1 * (target * logpt).sum()
        return loss
