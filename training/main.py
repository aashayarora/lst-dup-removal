from argparse import ArgumentParser
import json

import torch
torch.manual_seed(42)
import torch.optim as optim

from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

from model import DRNetwork
from dataset import GraphDataset
from train import train_batch, validate_batch
from loss import ContrastiveLoss

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

import numpy as np

from tqdm import tqdm

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load the configuration parameters
    input_path = config.get('input_path', '../data/')
    output_path = config.get('output_path', './')
    regex = config.get('regex', 'graph_*.pt')
    
    epochs = config.get('epochs', 50)
    validation_step = config.get('validation_step', 5)
    
    batch_size = config.get('batch_size', 16)
    
    hidden_dim = config.get('hidden_dim', 8)
    output_dim = config.get('output_dim', 32)

    knn_neighbors = config.get('knn_neighbors', 5)

    train_size = config.get('training_split', 0.8)
    learning_rate = config.get('learning_rate', 0.005)

    margin = config.get('CL_margin', 0.5)
    
    subset = config.get('subset', None)
    if args.debug:
        subset = 10

    dataset = GraphDataset(input_path=input_path, regex=regex, subset=subset)

    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, random_state=42)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16, drop_last=True, num_workers=4, pin_memory=True)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    num_node_features = dataset[0].num_node_features

    model = DRNetwork(
        input_dim=num_node_features,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        k=knn_neighbors
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=margin)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(1, epochs+1)):
        batch_loss_total = []
        batch_loss_attractive = []
        batch_loss_repulsive = []
        for data in train_loader:
            data = data.to(device)
            loss = train_batch(model, optimizer, criterion, data)
            batch_loss_total.append(loss[0].item())
            batch_loss_attractive.append(loss[1].item())
            batch_loss_repulsive.append(loss[2].item())
        avg_batch_loss = np.mean(batch_loss_total)
        avg_batch_loss_attractive = np.mean(batch_loss_attractive)
        avg_batch_loss_repulsive = np.mean(batch_loss_repulsive)
        train_losses.append([avg_batch_loss, avg_batch_loss_attractive, avg_batch_loss_repulsive])
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_batch_loss}")
            
        if epoch % validation_step == 0:
            batch_loss_total = []
            batch_loss_attractive = []
            batch_loss_repulsive = []
            for data in test_loader:
                data = data.to(device)
                loss = validate_batch(model, criterion, data)
                batch_loss_total.append(loss[0].item())
                batch_loss_attractive.append(loss[1].item())
                batch_loss_repulsive.append(loss[2].item())
            avg_batch_loss = np.mean(batch_loss_total)
            avg_batch_loss_attractive = np.mean(batch_loss_attractive)
            avg_batch_loss_repulsive = np.mean(batch_loss_repulsive)
            test_losses.append([avg_batch_loss, avg_batch_loss_attractive, avg_batch_loss_repulsive])
            tqdm.write(f"Epoch {epoch}, Test Loss: {avg_batch_loss}")

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    fig, ax = plt.subplots()
    ax.plot(train_losses[:, 0], label='Train Loss')
    ax.plot(train_losses[:, 1], label='Train Loss (Attractive)')
    ax.plot(train_losses[:, 2], label='Train Loss (Repulsive)')
    ax.plot(np.arange(validation_step, epochs+1, validation_step), test_losses[:, 0], label='Test Loss')
    ax.plot(np.arange(validation_step, epochs+1, validation_step), test_losses[:, 1], label='Test Loss (Attractive)')
    ax.plot(np.arange(validation_step, epochs+1, validation_step), test_losses[:, 2], label='Test Loss (Repulsive)')
    ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f"{output_path}/loss_plot.png")


    # Save the model
    torch.save(model.state_dict(), f"{output_path}/model.pt")
    print(f"Model saved to {output_path}/model.pt")