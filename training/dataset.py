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

    def __inc__(self, key, value):
        if key == 'edge_index' or key == 'pair_idxs_left' or key == 'pair_idxs_right':
            return self.num_nodes
        return super().__inc__(key, value)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False)
        # graph x: pt, eta, sinphi, cosphi, type, phi
        graph.x[:, 0] = torch.log10(1 + graph.x[:, 0])  # pt
        graph.x[:, 1] = graph.x[:, 1] / 5  # eta
        graph.x[:, 4] = graph.x[:, 4] / 10  # type
        if graph.x.shape[1] > 5:
            graph.x[:, 5] = graph.x[:, 5] / 3.14  # phi normalization
        return graph