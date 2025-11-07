import torch
import torch.nn as nn
import numpy as np
import LIMITR_loss

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class CELoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)

    def forward(self, output, target):
        '''
        Output: (N,*,C) \n
        Target: (N,*) \n
        '''
        output = torch.log(output)
        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1).long()
        return self.CELoss(output, target)

class CELossShift(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss(ignore_index=ignore_index)

    def forward(self, output, target):
        output = output[:,:-1,:]
        target = target[:,1:]
        return self.CELoss(output, target)

class CELossTotal(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss()
        self.CELossShift = CELossShift(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])



