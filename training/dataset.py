from torch_geometric.data import Dataset
import torch
from glob import glob

class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=None):
        super().__init__()
        self.subset = subset
        self.files = glob(input_path + regex)

    def __len__(self):
        if self.subset is not None:
            return min(self.subset, len(self.files))
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False)
        # graph x: pt, eta, sinphi, cosphi, type
        graph.x[:, 0] = graph.x[:, 0] / 10
        graph.x[:, 1] = graph.x[:, 1] / 5
        graph.x[:, 4] = graph.x[:, 4] / 10
        return graph