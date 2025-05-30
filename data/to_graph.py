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
    'tc_isDuplicate',
    'tc_matched_simIdx',
    'tc_lsIdx',
    'tc_t3Idx'
]

OUTPUT_FEATURES = [
    'tc_pt',
    'tc_eta',
    'tc_sinphi',
    'tc_cosphi',
    'tc_type'
]

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(input_path)["tree"]

    def process_event(self, arr, idx, debug=False, overwrite=False):
        """Processes a single event and saves the graph."""
        try:
            output_file = os.path.join(self.output_path, f"graph_{idx}.pt")
            if debug:
                ...
            elif ((not overwrite) and os.path.exists(output_file)):
                print(f"Graph {idx} already exists, skipping...")
                return
            
            duplicate_mask = arr["tc_isDuplicate"] == 1

            arr["tc_idx"] = [np.arange(len(arr["tc_pt"][0]))]
            dup_sim_idxs = ak.to_dataframe(ak.fill_none(ak.firsts(arr["tc_matched_simIdx"][duplicate_mask], axis=-1), -1))
            duplicate_mask = duplicate_mask[0]

            dup_pair_idxs = {}
            for i, simIdx in enumerate(dup_sim_idxs.values.flatten()):
                if simIdx == -1:
                    continue
                if simIdx not in dup_pair_idxs:
                    dup_pair_idxs[simIdx] = []
                dup_pair_idxs[simIdx].append(arr["tc_idx"][0][duplicate_mask][i])

            pair_idxs_left = []
            pair_idxs_right = []    

            dup_pair_labels = []
            for simIdx, pair_idxs in dup_pair_idxs.items():
                if len(pair_idxs) != 2:
                    continue
                pair_idxs_left.append(pair_idxs[0])
                pair_idxs_right.append(pair_idxs[1])
                dup_pair_labels.append(1)

            pts = arr["tc_pt"][0]
            etas = arr["tc_eta"][0]
            phis = arr["tc_phi"][0]
            n_particles = len(pts)

            i_indices, j_indices = np.tril_indices(n_particles, k=-1)

            both_duplicates = np.logical_and(duplicate_mask[i_indices], duplicate_mask[j_indices])
            valid_i = i_indices[~both_duplicates]
            valid_j = j_indices[~both_duplicates]

            delta_phi = np.abs(((phis[valid_i] - phis[valid_j] + np.pi) % (2 * np.pi)) - np.pi)

            dR2 =(etas[valid_i] - etas[valid_j])**2 + delta_phi**2

            close_pairs_mask = dR2 < 0.02
            close_i = valid_i[close_pairs_mask]
            close_j = valid_j[close_pairs_mask]

            if len(close_i) > 0:
                n_samples = min(len(pair_idxs_left), len(close_i))
                if n_samples > 0:
                    sample_indices = np.random.choice(len(close_i), size=n_samples, replace=False)
                    
                    sampled_i = close_i[sample_indices]
                    sampled_j = close_j[sample_indices]
                    
                    pair_idxs_left.extend(sampled_i.tolist())
                    pair_idxs_right.extend(sampled_j.tolist())
                    dup_pair_labels.extend([0] * n_samples)
            
            pair_idxs_left = np.array(pair_idxs_left, dtype=np.int64)
            pair_idxs_right = np.array(pair_idxs_right, dtype=np.int64)

            arr["tc_sinphi"] = np.sin(arr["tc_phi"])
            arr["tc_cosphi"] = np.cos(arr["tc_phi"])
            dup_features = ak.to_dataframe(arr[OUTPUT_FEATURES])

            edge_index = self._build_edges(arr, dup_features, pair_idxs_left, pair_idxs_right)

            graph = Data(
                x=torch.tensor(dup_features.values, dtype=torch.float),
                y=torch.tensor(dup_pair_labels, dtype=torch.float),
                edge_index=edge_index,
                pair_idxs_left=torch.tensor(pair_idxs_left, dtype=torch.long),
                pair_idxs_right=torch.tensor(pair_idxs_right, dtype=torch.long),
            )
            
            torch.save(graph, output_file)
            print(f"Processed graph {idx}")

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    @staticmethod
    def _build_edges(arr, dup_features, pair_idxs_left, pair_idxs_right):
        lsidxs = arr["tc_lsIdx"][0]  # Jagged array of lsIdx values
        
        # Initialize lists to store source and destination indices for edges
        edge_src = []
        edge_dst = []
        
        # Create a mapping from each lsIdx to the nodes that have it
        lsidx_to_nodes = {}
        
        # Handle awkward array conversion if needed
        if hasattr(lsidxs, 'tolist'):
            lsidxs_list = lsidxs.tolist()
        else:
            lsidxs_list = lsidxs
        
        # Populate the mapping
        for node_idx, node_lsidxs in enumerate(lsidxs_list):
            if node_lsidxs is None:
                continue
            for lsidx in node_lsidxs:
                if lsidx not in lsidx_to_nodes:
                    lsidx_to_nodes[lsidx] = []
                lsidx_to_nodes[lsidx].append(node_idx)
        
        # For each lsIdx, create edges between all pairs of nodes that share it
        for lsidx, nodes in lsidx_to_nodes.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Add edge in both directions (for undirected graph representation)
                    edge_src.append(nodes[i])
                    edge_dst.append(nodes[j])
                    edge_src.append(nodes[j])
                    edge_dst.append(nodes[i])
        
        # Convert to PyTorch tensor with shape [2, num_edges]
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        
        return edge_index

    def process_events_in_parallel(self, n_workers, debug=False, overwrite=False):
        num_events = self.input_tree.num_entries if not debug else 1
        n_workers =  min(n_workers, os.cpu_count() // 2) if not debug else 1

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx in range(num_events):
                executor.submit(
                    self.process_event,
                    self.input_tree.arrays(FEATURES, entry_start=idx, entry_stop=idx + 1),
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