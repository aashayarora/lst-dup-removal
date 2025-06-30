import torch.nn as nn
import torch
from torch_geometric.nn import SimpleConv, GATConv, GCNConv
    
class DRNetwork(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None):
        super().__init__()
        
        if input_dim is not None and hidden_dim is not None and output_dim is not None:
            self.linear = nn.Linear(input_dim, hidden_dim)
            # layers
            self.conv = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.embedding_dnn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.Linear(2 * hidden_dim, output_dim)
            )
    
    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
        )

    def forward(self, data):
        x = self.linear(data.x).relu()
        x = self.conv(x, edge_index=data.edge_index, edge_attr=data.edge_attr).relu()
        
        x = self.embedding_dnn(data.x)

        first_embeddings = x[data.pair_idxs_left]
        second_embeddings = x[data.pair_idxs_right]

        first_feature = data.x[data.pair_idxs_left]
        second_feature = data.x[data.pair_idxs_right]
        
        pair_embeddings = torch.stack([first_embeddings, second_embeddings])
        pair_features = torch.stack([first_feature, second_feature])
        
        return pair_embeddings, pair_features, data.y