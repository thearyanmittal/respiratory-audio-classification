import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    per_cls_weights = (1-beta)/(1-beta**torch.FloatTensor(cls_num_list))
    return per_cls_weights / sum(per_cls_weights) * len(cls_num_list)


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list, gamma=0., device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.weight = reweight(cls_num_list).to(self.device)

    def forward(self, input, target):
        input, target = input.to(self.device), target.to(self.device)
        pt = F.softmax(input)[torch.arange(len(target)), target]
        return torch.mean(torch.mul(torch.pow(torch.sub(1, pt), self.gamma), F.cross_entropy(input, target, weight=self.weight)))