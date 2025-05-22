import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, output, target):
        z1, z2 = output[0], output[1]
        dist_sq = torch.sum((z1 - z2)**2, dim=1) 
        dist = torch.sqrt(dist_sq + self.eps)

        attr = target*dist_sq
        rep  = (1 - target) * (F.relu(self.margin - dist)**2)

        loss  = torch.mean(attr + rep)
        return loss, torch.mean(attr), torch.mean(rep)