import torch.nn as nn
import torch
from torch_geometric.nn import SimpleConv, GATConv, GCNConv
from torch_cluster import knn_graph
    
class DRNetwork(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, k=None):
        super().__init__()
        self.k = k
        
        if input_dim is not None and hidden_dim is not None and output_dim is not None:
            self.linear = nn.Linear(input_dim, hidden_dim)
            # layers
            self.conv = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
            # self.conv = SimpleConv()
            self.embedding_dnn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
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
        x = self.linear(data.x).relu()
        x = self.conv(x, data.edge_index).relu()
        
        x = self.embedding_dnn(x)
        
        if hasattr(data, 'batch') and data.batch.max() > 0:  # We have multiple graphs in the batch
            # Get the offset for each node
            # Determine graph offsets (as you already have)
            batch_size = data.batch.max().item() + 1
            offset = torch.zeros(batch_size, dtype=torch.long, device=data.batch.device)
            _, counts = torch.unique(data.batch, return_counts=True)
            offset[1:] = torch.cumsum(counts[:-1], dim=0)

            # Create tensor to track which graph each pair belongs to
            num_pairs = data.pair_idxs_left.size(0)
            pair_batch = torch.zeros(num_pairs, dtype=torch.long, device=data.batch.device)

            # Process each graph separately to determine pair batch ownership
            for graph_idx in range(batch_size):
                # Find nodes belonging to this graph
                graph_mask = data.batch == graph_idx
                graph_nodes = torch.nonzero(graph_mask, as_tuple=True)[0]
                
                # Find pairs where both left and right indices are in this graph's node range
                min_idx, max_idx = graph_nodes.min(), graph_nodes.max()
                pair_mask = ((data.pair_idxs_left >= min_idx - offset[graph_idx]) & 
                            (data.pair_idxs_left <= max_idx - offset[graph_idx]) &
                            (data.pair_idxs_right >= min_idx - offset[graph_idx]) &
                            (data.pair_idxs_right <= max_idx - offset[graph_idx]))
                
                # Assign graph index to these pairs
                pair_batch[pair_mask] = graph_idx

            # Now use pair_batch for adjusting indices
            adjusted_left = data.pair_idxs_left + offset[pair_batch]
            adjusted_right = data.pair_idxs_right + offset[pair_batch]
            
            first_embeddings = x[adjusted_left]
            second_embeddings = x[adjusted_right]
        else:
            # No batching or single graph, use indices directly
            first_embeddings = x[data.pair_idxs_left]
            second_embeddings = x[data.pair_idxs_right]
        
        pair_embeddings = torch.stack([first_embeddings, second_embeddings])
        
        return pair_embeddings, data.y