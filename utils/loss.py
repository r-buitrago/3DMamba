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
        target: (B, ) (class indices)
        '''
        # target comes as one-hot, convert to class
        target = target.argmax(dim=-1)
        B, C = input.shape
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        # Gather the probabilities corresponding to the target classes
        logpt = logpt.gather(1, target.view(-1, 1))
        pt = pt.gather(1, target.view(-1, 1))
        logpt = (1-pt)**self.gamma * logpt
        loss = -1 * logpt.sum()
        return loss

class CrossEntropyBatchWeighted(nn.Module):
    def __init__(self):
        super(CrossEntropyBatchWeighted, self).__init__()
    
    def forward(self, input, target):
        ''' input: (B, num_classes)
        target: (B, num_classes) (one-hot)
        '''
        target_frequencies = target.sum(dim=0)
        weights = torch.zeros_like(target_frequencies)
        weights[target_frequencies > 0] = 1 / target_frequencies[target_frequencies > 0]
        weights = weights / weights.sum()
        # target comes as one-hot, convert to class
        target = target.argmax(dim=-1)
        B, C = input.shape
        # count number of appareances of each class in target
        # weighted softmax
        return F.cross_entropy(input, target, weight=weights, reduction="sum")
        