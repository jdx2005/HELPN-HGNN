import os
import argparse
import numpy as np
import random
import nibabel as nib
from nibabel.affines import apply_affine
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import threshold_otsu
from scipy.stats import zscore
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, JumpingKnowledge
from torch.utils.data import random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import time
import logging

def load_fmri(path):
    img = nib.load(path)
    data = img.get_fdata()  # shape (X, Y, Z, T)
    affine = img.affine
    return data, affine

def extract_superpixels(mean_vol, n_segments=500, compactness=0.1):
    segments = slic(
        mean_vol,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        start_label=0
    )
    return segments

def compute_region_timeseries(data, segments):
    T = data.shape[-1]
    labels = np.unique(segments)
    ts_list = []
    for lbl in labels:
        mask = (segments == lbl)
        ts = zscore(data[mask, ...].reshape(-1, T), axis=1)
        ts_mean = np.nanmean(ts, axis=0)
        ts_list.append(ts_mean)
    return np.vstack(ts_list), labels


def select_dynamic_regions(timeseries, pctl=90):
    variances = np.var(timeseries, axis=1)
    thr = np.percentile(variances, pctl)
    return np.where(variances >= thr)[0]

def build_correlation_graph(ts_sel, labels_sel, corr_thr=0.7):
    corr = np.corrcoef(ts_sel)
    G = nx.Graph()
    for i, lbl in enumerate(labels_sel):
        G.add_node(int(lbl), variance=np.var(ts_sel[i]))
    for i in range(len(labels_sel)):
        for j in range(i + 1, len(labels_sel)):
            if corr[i, j] >= corr_thr:
                G.add_edge(int(labels_sel[i]), int(labels_sel[j]), weight=corr[i, j])
    return G, corr

def extract_smri_feats(path):
    """
    Load sMRI NIfTI (3D), take mid-axis slice, compute with Otsu segmentation:
      1) Brain tissue proportion (proportion of pixels above threshold in slice)
      2) Mean brain tissue intensity
    """
    img = nib.load(path)
    data = img.get_fdata()             # shape (X, Y, Z)
    mid_z = data.shape[2] // 2
    slice2d = data[:, :, mid_z]
    thr = threshold_otsu(slice2d)
    mask = slice2d > thr
    prop = mask.sum() / mask.size
    mean_int = float(slice2d[mask].mean()) if mask.any() else 0.0
    return np.array([prop, mean_int], dtype=np.float32)


# Added: Load cached graph data
def load_cached_graphs(cache_dir):
    data_list = []
    if not os.path.isdir(cache_dir):
        return data_list
    for fn in sorted(os.listdir(cache_dir)):
        if fn.endswith('.pt'):
            fp = os.path.join(cache_dir, fn)
            try:
                # Disable weights_only to allow loading complete Data, including edge_attr and other custom fields
                data_list.append(torch.load(fp, weights_only=False))
            except Exception as e:
                print(f"[Skip corrupted cache] {fp} : {e}")
    return data_list

def parse_args():
    parser = argparse.ArgumentParser(description="fMRI graph structure + hyper-events + sMRI GNN binary classification")
    parser.add_argument('--data_dir',  required=True,
                        help="Root directory containing 0/1 subfolders, each with subject directories")
    parser.add_argument('--epochs',    type=int,   default=100)
    parser.add_argument('--batch_size',type=int,   default=8)
    parser.add_argument('--lr',        type=float, default=5e-4)
    parser.add_argument('--val_size',  type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--dropout',   type=float, default=0.7, help='Dropout probability')
    # Added cache-related parameters
    parser.add_argument('--graph_cache_dir', type=str, default='graph_cache',
                        help='Directory to save/load graph structure cache (.pt)')
    parser.add_argument('--use_cached', action='store_true',
                        help='If specified, train directly from cached graph structures without regenerating')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dim',     type=int,   default=128,
                        help='GNN hidden layer dimension')
    parser.add_argument('--gat_heads',      type=int,   default=16,
                        help='Number of GAT heads')
    parser.add_argument('--weight_decay',   type=float, default=1e-4,
                        help='Optimizer L2 regularization coefficient')
    parser.add_argument('--scheduler',      choices=['plateau','cosine'],
                        default='plateau', help='Learning rate scheduler type')
    parser.add_argument('--patience',       type=int,   default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--label_smoothing',type=float, default=0.1,
                        help='Cross-entropy label smoothing (0 to disable)')
    return parser.parse_args()

def main():
    start_time = time.time()
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler("pt_generation_time.log", mode="a")  # Output to file
        ]
    )

    args = parse_args()
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_list = []
    if args.use_cached:
        data_list = load_cached_graphs(args.graph_cache_dir)
        print(f"Read graph samples from cache: {len(data_list)}")
        if len(data_list) == 0:
            print("Cache is empty or does not exist, switching to regenerating graph structures.")
            args.use_cached = False
        else:
            # Added: Check if cache is complete, if not, supplement generation
            total_expected = 0
            for dataset in sorted(os.listdir(args.data_dir)):
                dataset_dir = os.path.join(args.data_dir, dataset)
                if not os.path.isdir(dataset_dir):
                    continue
                for lbl in ['0','1']:
                    class_dir = os.path.join(args.data_dir, 'fMRI', lbl)
                    if not os.path.isdir(class_dir):
                        continue
                    for subj in sorted(os.listdir(class_dir)):
                        # fMRI path
                        sd_fmri   = os.path.join(class_dir, subj)
                        fmri_path = os.path.join(sd_fmri, 'rest_1', 'rest.nii.gz')
                        # sMRI path
                        smri_path = os.path.join(
                            args.data_dir, 'sMRI', lbl, subj,
                            'anat_1', 'anat.nii.gz'
                        )
                        if os.path.exists(smri_path) and os.path.exists(fmri_path):
                            total_expected += 1
            if len(data_list) < total_expected:
                print(f"[INFO] Cached graph count {len(data_list)} is less than valid sample count {total_expected}, will supplement missing graph structures.")
                args.use_cached = False

    if not args.use_cached:
        os.makedirs(args.graph_cache_dir, exist_ok=True)
        # Get set of cached graph filenames
        cached = {fn for fn in os.listdir(args.graph_cache_dir) if fn.endswith('.pt')}
        for dataset in sorted(os.listdir(args.data_dir)):
            dataset_dir = os.path.join(args.data_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue
            for lbl in ['0','1']:
                # fMRI and sMRI base directories
                fMRI_dir = os.path.join(dataset_dir, 'fMRI', lbl)
                sMRI_dir = os.path.join(dataset_dir, 'sMRI', lbl)
                if not os.path.isdir(fMRI_dir) or not os.path.isdir(sMRI_dir):
                    continue
                for subj in sorted(os.listdir(fMRI_dir)):
                    save_name = f"{lbl}_{subj}.pt"
                    if save_name in cached:
                        continue
                    # Build respective input paths
                    fmri_path = os.path.join(fMRI_dir, subj,  'rest_1', 'NIfTI', 'rest.nii.gz')
                    smri_path = os.path.join(sMRI_dir, subj,  'anat_1', 'NIfTI', 'rest.nii.gz')
                    # print(fmri_path)
                    if not os.path.exists(fmri_path) or not os.path.exists(smri_path):
                        continue
                    try:
                        data4d, affine = load_fmri(fmri_path)
                    except Exception as e:
                        print(f"[Skip] Failed to read fMRI: {fmri_path} => {e}")
                        continue
                    mean_vol = data4d.mean(axis=-1)
                    segments = extract_superpixels(mean_vol)
                    ts, labels_seg = compute_region_timeseries(data4d, segments)
                    idx_dyn = select_dynamic_regions(ts)
                    ts_sel, lbl_sel = ts[idx_dyn], labels_seg[idx_dyn]
                    G, _ = build_correlation_graph(ts_sel, lbl_sel)
                    if len(G.nodes()) == 0 or len(G.edges()) == 0:
                        continue
                    node_labels = list(G.nodes())
                    label2idx = {old: i for i, old in enumerate(node_labels)}
                    node_feats = torch.tensor(
                        [G.nodes[n]['variance'] for n in node_labels],
                        dtype=torch.float
                    ).unsqueeze(1)
                    edge_tuples = [(label2idx[u], label2idx[v]) for u, v in G.edges()]
                    if len(edge_tuples) == 0:
                        continue
                    edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
                    edge_weight= torch.tensor(
                        [G[u][v]['weight'] for u,v in G.edges()],
                        dtype=torch.float
                    )
                    smri_feat = torch.tensor(extract_smri_feats(smri_path), dtype=torch.float).unsqueeze(0)
                    data_obj = Data(
                        x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_weight,
                        y=torch.tensor([int(lbl)], dtype=torch.long),
                        smri_feat=smri_feat
                    )
                    data_list.append(data_obj)
                    # Save to cache
                    torch.save(data_obj, os.path.join(args.graph_cache_dir, save_name))
        print(f"Generated and cached graph samples: {len(data_list)} to {args.graph_cache_dir}")

    print(f"Total samples loaded: {len(data_list)}")
    if len(data_list) == 0:
        print("No valid samples, exiting.")
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"Graph structure generation completed, elapsed time {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
