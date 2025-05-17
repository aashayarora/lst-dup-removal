import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
from torch_cluster import knn_graph
    
class DRNetwork(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, k=None):
        super().__init__()
        self.k = k
        
        if input_dim is not None and hidden_dim is not None and output_dim is not None:
            self.linear = nn.Linear(input_dim, hidden_dim)
            # layers
            self.conv = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.embedding_dnn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, output_dim)
            )
    
    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            k=config["knn_neighbors"]
        )

    def forward(self, data):
        x = self.linear(data.x)
        edge_index = knn_graph(x, batch=data.batch, k=self.k)
        x = self.conv(x, edge_index)
        
        x = self.embedding_dnn(x)

        # Use the precomputed pairs
        if data.pairs_indices.shape[0] > 0:
            # Get batch-specific offsets for each node
            batch_size = data.batch.max().item() + 1
            
            # Calculate cumulative counts to get offsets (faster approach)
            ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=data.batch.device)
            unique, counts = torch.unique(data.batch, sorted=True, return_counts=True)
            ptr[1:][unique] = counts
            ptr = ptr.cumsum(dim=0)
            
            # Get batch information for each pair (first node in the pair)
            batch_for_pairs = data.batch_for_pairs if hasattr(data, 'batch_for_pairs') else data.batch[data.pairs_indices[:, 0]]
            
            # Vectorized approach: get the offset for each batch index
            offsets = ptr[batch_for_pairs]
            
            # Apply offsets to both nodes in each pair (vectorized)
            adjusted_first_indices = data.pairs_indices[:, 0] - offsets
            adjusted_second_indices = data.pairs_indices[:, 1] - offsets
            
            # Get embeddings for the pairs
            first_embeddings = x[data.pairs_indices[:, 0]]
            second_embeddings = x[data.pairs_indices[:, 1]]
            
            # Stack pairs for contrastive loss
            pair_embeddings = torch.stack([first_embeddings, second_embeddings])
            
            return pair_embeddings, data.pairs_labels
        else:
            # Return empty tensors if no pairs are available
            return torch.zeros((2, 0, x.size(1)), device=x.device), torch.zeros(0, device=x.device)