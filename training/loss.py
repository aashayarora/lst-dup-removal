import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        # Compute the pairwise distance
        dist = torch.sqrt(torch.sum((output[0] - output[1]) ** 2))
        # Compute the loss
        attractive_loss = 5 * torch.mean((target) * dist ** 2)
        repulsive_loss = torch.mean((1 - target) * torch.clamp(self.margin - dist, min=0) ** 2)
        loss = torch.mean(attractive_loss + repulsive_loss)
        return loss, attractive_loss, repulsive_loss