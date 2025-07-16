import os
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt

FEATURES = [
    'tc_pt',
    'tc_eta',
    'tc_phi',
    'tc_type',
    'tc_isDuplicate',
    'tc_matched_simIdx',
    'tc_lsIdx',
]

OUTPUT_FEATURES = [
    'tc_pt',
    'tc_eta',
    'tc_phi',
    'tc_sinphi',
    'tc_cosphi',
    'tc_type',
]

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return dphi

def delta_R(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

class Graph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index' or key == 'pair_idxs_left' or key == 'pair_idxs_right':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

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
            
            tc_idx = np.arange(len(arr["tc_eta"][0]))
            duplicate_mask = (arr["tc_isDuplicate"] == 1)[0]

            tc_matched_simidx = ak.firsts(ak.where(ak.num(arr["tc_matched_simIdx"][0]) == 0, [[-1]] * len(arr["tc_matched_simIdx"][0]), arr["tc_matched_simIdx"][0]))
            dup_sim_idxs = tc_matched_simidx[duplicate_mask]
            
            dup_pair_idxs = {}
            for i, simIdx in enumerate(dup_sim_idxs):
                if simIdx == -1:
                    continue
                if simIdx not in dup_pair_idxs:
                    dup_pair_idxs[simIdx] = []
                dup_pair_idxs[simIdx].append(tc_idx[duplicate_mask][i])

            pair_idxs_left = []
            pair_idxs_right = []    

            for simIdx, pair_idxs in dup_pair_idxs.items():
                if len(pair_idxs) < 2:
                    continue
                pair_idxs_left.append(pair_idxs[0])
                pair_idxs_right.append(pair_idxs[1])

            pts = arr["tc_pt"][0]
            etas = arr["tc_eta"][0]
            phis = arr["tc_phi"][0]

            pair_idxs_left = np.array(pair_idxs_left)
            pair_idxs_right = np.array(pair_idxs_right)

            deltaR_dups = delta_R(etas[np.array(pair_idxs_left)], phis[np.array(pair_idxs_left)], etas[np.array(pair_idxs_right)], phis[np.array(pair_idxs_right)])

            # Remove pairs with deltaR == 0 (exact duplicates)
            pair_idxs_left = list(pair_idxs_left[deltaR_dups > 0])
            pair_idxs_right = list(pair_idxs_right[deltaR_dups > 0])
            dup_pair_labels = [1] * len(pair_idxs_left)

            # NON-DUPLICATES
            n_particles = len(pts)

            i_indices, j_indices = np.tril_indices(n_particles, k=-1)

            both_duplicates = np.logical_and(duplicate_mask[i_indices], duplicate_mask[j_indices])
            valid_i = i_indices[~both_duplicates]
            valid_j = j_indices[~both_duplicates]
            
            deltaR_nodup = delta_R(etas[valid_i], phis[valid_i], etas[valid_j], phis[valid_j])

            # Add diverse negative samples instead of just close pairs
            negative_pairs_i = []
            negative_pairs_j = []
            negative_labels = []

            close_pairs_mask = (deltaR_nodup > 0.) & (deltaR_nodup < 0.1)
            close_i = valid_i[close_pairs_mask]
            close_j = valid_j[close_pairs_mask]

            # 1. Sample some close but non-duplicate pairs (hard negatives)
            if len(close_i) > 0:
                negative_pairs_i.extend(close_i)
                negative_pairs_j.extend(close_j)
                negative_labels.extend([0] * len(close_i))

            # Combine all pairs
            pair_idxs_left.extend(negative_pairs_i)
            pair_idxs_right.extend(negative_pairs_j)
            dup_pair_labels.extend(negative_labels)

            # Ensure we have some examples (skip if no valid pairs)
            if len(pair_idxs_left) == 0:
                print(f"No valid pairs found for event {idx}, skipping...")
                return

            arr["tc_sinphi"] = np.sin(arr["tc_phi"])
            arr["tc_cosphi"] = np.cos(arr["tc_phi"])
            dup_features = ak.to_dataframe(arr[OUTPUT_FEATURES])

            edge_index, edge_attr = self._build_edges(arr, dup_features)

            graph = Graph(
                x=torch.tensor(dup_features.values, dtype=torch.float),
                y=torch.tensor(dup_pair_labels, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pair_idxs_left=torch.tensor(pair_idxs_left, dtype=torch.long),
                pair_idxs_right=torch.tensor(pair_idxs_right, dtype=torch.long),
            )
            torch.save(graph, output_file)
            print(f"Processed graph {idx}")

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    @staticmethod
    def _build_edges(arr, dup_features):
        lsidxs = arr["tc_lsIdx"][0]  # Jagged array of lsIdx values
        etas = arr["tc_eta"][0]
        phis = arr["tc_phi"][0]
        
        # Initialize lists to store source and destination indices for edges
        edge_src = []
        edge_dst = []
        edge_attrs = []
        
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
                    node_i, node_j = nodes[i], nodes[j]
                    
                    # Calculate dR² between the two nodes
                    deltaR = delta_R(etas[node_i], phis[node_i], etas[node_j], phis[node_j])

                    # Add edge in both directions (for undirected graph representation)
                    edge_src.extend([node_i, node_j])
                    edge_dst.extend([node_j, node_i])
                    edge_attrs.extend([deltaR, deltaR])
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)  # Shape [num_edges, 1]
        
        return edge_index, edge_attr

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