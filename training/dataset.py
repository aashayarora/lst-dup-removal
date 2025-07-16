from torch_geometric.data import Data, Dataset
import torch
from glob import glob

class Graph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index' or key == 'pair_idxs_left' or key == 'pair_idxs_right':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
    
class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=None):
        super().__init__()
        self.subset = subset
        self.files = glob(input_path + regex)
        print(f"Found {len(self.files)} files matching {regex} in {input_path}")

    def __len__(self):
        if self.subset is not None:
            return min(self.subset, len(self.files))
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False)

        # Feature normalization
        graph.x[:, 0] = torch.log10(1 + graph.x[:, 0])  # pt
        graph.x[:, 1] = graph.x[:, 1] / 5  # eta
        graph.x[:, 2] = graph.x[:, 2] / 3.14  # phi
        graph.x[:, 5] = graph.x[:, 5] / 10  # type

        return graph