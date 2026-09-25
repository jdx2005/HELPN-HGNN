import os
import re
import json
import logging

import torch
import pandas as pd

logger = logging.getLogger(__name__)


def extract_subject_id(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return None


def load_cached_graphs(cache_dir):
    data_list = []
    if not os.path.isdir(cache_dir):
        return data_list
    for fn in sorted(os.listdir(cache_dir)):
        if fn.endswith('.pt'):
            fp = os.path.join(cache_dir, fn)

            subject_id = extract_subject_id(fn)
            if not subject_id:
                logger.warning(f"Skipping {fn} as subject ID could not be extracted.")
                continue

            # Derive the corresponding .json file name from the .pt file name
            prefix = fn.split('_')[1]
            json_fp = os.path.join(cache_dir, f"{prefix}_{subject_id}_1_hypergraph.json")

            if not os.path.isfile(json_fp):
                logger.warning(f"Skipping {fn} as corresponding JSON file {json_fp} is missing.")
                continue

            try:
                data = torch.load(fp, weights_only=False)
                data.name = fn

                with open(json_fp, 'r') as f:
                    hypergraph_data = json.load(f)

                hyperedges = hypergraph_data.get('hyperedge', {})
                weights = hypergraph_data.get('weights', {})

                edge_index = []
                edge_attr = []
                batch = []

                # Node deduplication: map node coordinates to unique IDs
                node_map = {}
                node_counter = 0

                for edge, nodes in hyperedges.items():
                    edge_id = int(re.search(r'\d+', edge).group())
                    subject_id_int = int(re.search(r'\d+', subject_id).group())
                    weight = weights.get(edge, 1.0)
                    for node in nodes:
                        if isinstance(node, list) and len(node) == 3:
                            node_tuple = tuple(node)
                            if node_tuple not in node_map:
                                node_map[node_tuple] = node_counter
                                node_counter += 1
                            node_id = node_map[node_tuple]
                            edge_index.append([edge_id, node_id])
                            edge_attr.append(node + [weight])
                            batch.append(subject_id_int)
                        else:
                            logger.warning(f"Invalid node format in edge {edge}: {node}")

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                batch = torch.tensor(batch, dtype=torch.long)

                data.hypergraph = {
                    'edge_index': edge_index,
                    'edge_attr': edge_attr,
                    'batch': batch
                }

                data_list.append(data)
            except Exception as e:
                logger.warning(f"[Skipping corrupted cache] {fp} : {e}")
    return data_list


def load_subject_features(excel_path):
    df = pd.read_excel(excel_path, header=None)
    gender_map = {'M': 0, '男': 0, 'F': 1, '女': 1}
    handed_map = {'R': 0, '右': 0, 'L': 1, '左': 1}
    features = {}
    for idx, row in df.iterrows():
        subj = str(row[0]).strip()
        gender = gender_map.get(str(row[1]).strip(), 0)
        handed = handed_map.get(str(row[2]).strip(), 0)
        features[subj] = [gender, handed]
    return features