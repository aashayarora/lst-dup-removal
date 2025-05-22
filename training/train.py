import torch
import numpy as np

def train_batch(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()

    pair_embeddings, labels = model(data)

    if labels.numel() == 0:
        return None
    
    loss = criterion(pair_embeddings, labels)
    loss[0].backward()
    optimizer.step()
    return loss

def validate_batch(model, criterion, data):
    model.eval()
    with torch.no_grad():
        pair_embeddings, labels = model(data)
        if labels.numel() == 0:
            return None
        loss = criterion(pair_embeddings, labels)
    return loss
