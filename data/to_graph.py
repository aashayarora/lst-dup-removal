import os
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Data

FEATURES = [
    'tc_pt',
    'tc_eta',
    'tc_phi',
    'tc_type',
]

TARGETS = [
    "tc_matched_simIdx"
]

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(input_path)["tree"]

    def process_event(self, event_data, idx, debug=False, overwrite=False):
        """Processes a single event and saves the graph."""
        try:
            output_file = os.path.join(self.output_path, f"graph_{idx}.pt")
            if debug:
                ...
            elif ((not overwrite) and os.path.exists(output_file)):
                print(f"Graph {idx} already exists, skipping...")
                return
            
            arr = event_data[FEATURES]
            arr["tc_sinphi"] = np.sin(arr["tc_phi"])
            arr["tc_cosphi"] = np.cos(arr["tc_phi"])
            arr = arr[["tc_pt", "tc_eta", "tc_sinphi", "tc_cosphi", "tc_type"]]

            node_features = torch.Tensor(ak.to_dataframe(arr).values)
            target = torch.Tensor(ak.to_dataframe(ak.fill_none(ak.firsts(event_data[TARGETS], axis=-1), -1)).values)

            # Extract simulation indices
            sim_indices = target.squeeze().numpy()
            num_nodes = node_features.shape[0]
            
            # Group nodes by simulation index (except for unmatched nodes with sim_idx == -1)
            sim_idx_to_nodes = {}
            for i in range(num_nodes):
                sim_idx = int(sim_indices[i])
                if sim_idx >= 0:  # Skip unmatched nodes
                    if sim_idx not in sim_idx_to_nodes:
                        sim_idx_to_nodes[sim_idx] = []
                    sim_idx_to_nodes[sim_idx].append(i)
            
            # Create mapping for duplicates - nodes with same simulation indices
            duplicate_mapping = {}
            pairs_indices = []
            pairs_labels = []
            
            # Create all possible pairs of duplicates
            for sim_idx, nodes in sim_idx_to_nodes.items():
                if len(nodes) >= 2:  # At least 2 nodes to form a pair
                    # Create all possible pairs from nodes with the same sim_idx
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            src_idx, dst_idx = nodes[i], nodes[j]
                            duplicate_mapping[src_idx] = dst_idx
                            duplicate_mapping[dst_idx] = src_idx
                            pairs_indices.append([src_idx, dst_idx])
                            pairs_labels.append(1)  # 1 means same particle (positive pair)
            
            # Count how many positive pairs we have
            num_pos_pairs = len(pairs_indices)
            
            # Generate an equal number of negative pairs (non-duplicates)
            if num_pos_pairs > 0:
                # Create a list of all possible node pairs
                all_nodes = list(range(num_nodes))
                
                # Sample negative pairs
                neg_pairs_count = 0
                max_attempts = num_pos_pairs * 10  # Prevent infinite loop
                attempts = 0
                
                while neg_pairs_count < num_pos_pairs and attempts < max_attempts:
                    # Randomly select two nodes
                    i, j = np.random.choice(all_nodes, 2, replace=False)
                    
                    # Check if they're not duplicates of each other
                    if i not in duplicate_mapping or duplicate_mapping[i] != j:
                        # Make sure this pair hasn't been added already
                        if [i, j] not in pairs_indices and [j, i] not in pairs_indices:
                            pairs_indices.append([i, j])
                            pairs_labels.append(0)  # 0 means different particles (negative pair)
                            neg_pairs_count += 1
                    
                    attempts += 1
            
            # Convert to tensors
            if pairs_indices:
                pairs_indices = torch.tensor(pairs_indices)
                pairs_labels = torch.tensor(pairs_labels)
            else:
                pairs_indices = torch.zeros((0, 2), dtype=torch.long)
                pairs_labels = torch.zeros(0, dtype=torch.long)
            
            # Convert duplicate mapping to PyTorch tensors
            duplicate_idx = torch.ones(num_nodes, dtype=torch.long) * -1  # Default: no duplicate (-1)
            for src_idx, dst_idx in duplicate_mapping.items():
                duplicate_idx[src_idx] = dst_idx
            
            # Add all metadata to the graph
            graph = Data(
                x=node_features, 
                y=target,
                pairs_indices=pairs_indices,
                pairs_labels=pairs_labels,
                duplicate_idx=duplicate_idx,
                batch_for_pairs=torch.zeros(pairs_indices.shape[0], dtype=torch.long) if pairs_indices.shape[0] > 0 else torch.zeros(0, dtype=torch.long)
            )

            if debug:
                print(graph)
                print(f"Total nodes: {num_nodes}")
                print(f"Duplicate pairs found: {num_pos_pairs}")
                print(f"Total contrastive pairs: {len(pairs_indices)}")
                return

            torch.save(graph, output_file)
            print(f"Processed graph {idx}")
            
        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    def process_events_in_parallel(self, n_workers, debug=False, overwrite=False):
        num_events = self.input_tree.num_entries if not debug else 1
        n_workers =  min(n_workers, os.cpu_count() // 2) if not debug else 1

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx in range(num_events):
                executor.submit(
                    self.process_event,
                    self.input_tree.arrays(FEATURES + TARGETS, entry_start=idx, entry_stop=idx + 1),
                    idx, debug, overwrite
                )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")

    args = argparser.parse_args()

    data = GraphBuilder(args.input, args.output)
    data.process_events_in_parallel(args.n_workers, args.debug, args.overwrite)