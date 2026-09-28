import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  
import argparse
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer  
import torch.nn.functional as F
from torch.utils.data import random_split 
from torch_geometric.nn import GATConv, JumpingKnowledge, HypergraphConv, GraphNorm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import logging  
import json 
import re
import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
    category=UserWarning
)

# Set up logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
fh = logging.FileHandler('run.log', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

try:
    from torch_geometric.nn.models.positional_encoding import LaplacianPE
except ImportError:
    class LaplacianPE(nn.Module):
        def __init__(self, k, model_dim):
            super().__init__()
            self.k, self.model_dim = k, model_dim
        def forward(self, edge_index=None, batch=None):
            n = batch.size(0) if batch is not None else 0
            device = batch.device if batch is not None else torch.device('cpu')
            return torch.zeros((n, self.model_dim), device=device)

# Define utility functions for data loading and preprocessing
def load_cached_graphs(cache_dir):
    data_list = []
    if not os.path.isdir(cache_dir):
        return data_list
    for fn in sorted(os.listdir(cache_dir)):
        if fn.endswith('.pt'):
            fp = os.path.join(cache_dir, fn)
            
            # Extract subject ID and other key information
            subject_id = extract_subject_id(fn)
            if not subject_id:
                logger.warning(f"Skipping {fn} as subject ID could not be extracted.")
                continue
            
            # Dynamically generate corresponding .json filename based on .pt filename
            prefix = fn.split('_')[1]  # Extract prefix (e.g., KKI, Peking, etc.)
            json_fp = os.path.join(cache_dir, f"{prefix}_{subject_id}_1_hypergraph.json")
            
            # Check if corresponding .json file exists
            if not os.path.isfile(json_fp):
                logger.warning(f"Skipping {fn} as corresponding JSON file {json_fp} is missing.")
                continue
            
            try:
                # Load .pt file
                data = torch.load(fp, weights_only=False)
                data.name = fn
                
                # Load .json file
                with open(json_fp, 'r') as f:
                    hypergraph_data = json.load(f)
                
                # Process hyperedges and weights
                hyperedges = hypergraph_data.get('hyperedge', {})
                weights = hypergraph_data.get('weights', {})
                
                # Build edge_index and edge_attr
                edge_index = []
                edge_attr = []
                batch = []

                # Node deduplication dictionary: map node coordinates to unique IDs
                node_map = {}
                node_counter = 0  # Node ID counter

                for edge, nodes in hyperedges.items():
                    edge_id = int(re.search(r'\d+', edge).group())  # Extract first number and convert to integer
                    subject_id_int = int(re.search(r'\d+', subject_id).group())  # Extract numeric part of subject ID
                    weight = weights.get(edge, 1.0)  # Get weight, default value 1.0
                    for node in nodes:
                        if isinstance(node, list) and len(node) == 3:
                            # Convert node coordinates to tuple (immutable type, suitable as dictionary key)
                            node_tuple = tuple(node)
                            if node_tuple not in node_map:
                                # If node coordinates have not appeared, assign new node ID
                                node_map[node_tuple] = node_counter
                                node_counter += 1
                            # Get node ID
                            node_id = node_map[node_tuple]
                            # Store edge and node coordinates in edge_index
                            edge_index.append([edge_id, node_id])
                            # Store edge and weight in edge_attr
                            edge_attr.append(node + [weight])  # Edge weight
                            batch.append(subject_id_int)  # Assume subject_id is the current subject's ID
                        else:
                            logger.warning(f"Invalid node format in edge {edge}: {node}")
                # print(f"Constructed edge_index with {len(edge_index)} entries for {fn}")
                # Convert to PyTorch tensors
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Node coordinates
                # print("Edge index shape:", edge_index.shape)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Edge weights
                # print("Edge attr shape:", edge_attr.shape)
                batch = torch.tensor(batch, dtype=torch.long)  # Batch information indicating which graph each node belongs to
                # print("Batch shape:", batch.shape)
                # print(f"Loaded hypergraph for {fn}: edge_index shape {edge_index.shape}, edge_attr shape {edge_attr.shape}, batch shape {batch.shape}")
                
                # Attach hypergraph and weight information to data object
                data.hypergraph = {
                    'edge_index': edge_index,
                    'edge_attr': edge_attr,
                    'batch': batch
                }
                
                # Add data to list
                data_list.append(data)
            except Exception as e:
                logger.warning(f"[Skipping corrupted cache] {fp} : {e}")
    return data_list

def load_subject_features(excel_path):
    df = pd.read_excel(excel_path, header=None)
    gender_map = {'M':0, '男':0, 'F':1, '女':1}
    handed_map = {'R':0, '右':0, 'L':1, '左':1}
    features = {}
    for idx, row in df.iterrows():
        subj = str(row[0]).strip()
        gender = gender_map.get(str(row[1]).strip(), 0)
        handed = handed_map.get(str(row[2]).strip(), 0)
        features[subj] = [gender, handed]
    return features

def extract_subject_id(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return None

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="fMRI Graph Structure + Hyper-events + sMRI GNN Binary Classification")
    parser.add_argument('--data_dir',  required=True,
                        help="Root directory containing 0/1 subfolders, each with subject directories")
    parser.add_argument('--excel',     type=str,   required=True,
                        help='Path to Excel file with subject gender and handedness features')
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--batch_size',type=int,   default=16)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--val_size',  type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--dropout',   type=float, default=0.15, help='Dropout probability')
    parser.add_argument('--graph_cache_dir', type=str, default='graph_cache',
                        help='Directory to save/load graph structure cache (.pt)')
    parser.add_argument('--use_cached', action='store_true',
                        help='If specified, train directly from cached graph structures without regenerating')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dim',     type=int,   default=64,
                        help='GNN hidden layer dimension')
    parser.add_argument('--gat_heads',      type=int,   default=4,
                        help='Number of GAT heads')
    parser.add_argument('--weight_decay',   type=float, default=1e-4,
                        help='Optimizer L2 regularization coefficient')
    parser.add_argument('--scheduler',      choices=['plateau','cosine'],
                        default='plateau', help='Type of learning rate scheduler')
    parser.add_argument('--patience',       type=int,   default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--label_smoothing',type=float, default=0.1,
                        help='Cross-entropy label smoothing (0 to disable)')
    parser.add_argument('--num_prompts', type=int, default=3,
                        help='Number of prompt nodes injected per sample')
    parser.add_argument('--moe_experts', type=int, default=4,
                        help='Number of MoE experts')
    parser.add_argument('--moe_k', type=str, default="2",
                        help="List of selectable expert counts, comma-separated, e.g., '1,2,3'")
    parser.add_argument('--moe_thresh',  type=float, default=0.8,
                        help="MoE expert activation threshold")
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='LR warm-up epoch count')
    parser.add_argument('--class_weights', type=str, default='1.0,1.0',
                        help='Comma-separated class weights for imbalanced data handling')
    return parser.parse_args()

# Graph augmentation functions
def drop_edge(data, drop_prob=0.2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges, device=data.edge_index.device) > drop_prob
        data_aug = data.clone()
        data_aug.edge_index = data.edge_index[:, mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data_aug.edge_attr = data.edge_attr[mask]
        return data_aug
    return data

def add_noise_to_features(data, noise_std=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if hasattr(data, 'x') and data.x is not None:
        noise = torch.randn_like(data.x) * noise_std
        data_aug = data.clone()
        data_aug.x = data.x + noise
        return data_aug
    return data

def augment_graph(data, drop_prob=0.2, noise_std=0.1, seed=None):
    if(drop_prob > 0):
        data = drop_edge(data, drop_prob=drop_prob, seed=seed)
    if(noise_std > 0):
        data = add_noise_to_features(data, noise_std=noise_std, seed=seed)
    return data

# Define custom loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight,
                             label_smoothing=self.smoothing,
                             reduction='none')
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma)*ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class HypergraphCapsule(nn.Module):
    def __init__(self, out_channels):
        """
        Initialize the hypergraph capsule network module.
        Args:
        - out_channels: Dimension of output features.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hgconv = None  # Lazy initialization of HypergraphConv

    def forward(self, hyper_edge_index, hyper_edge_attr, hyper_batch):
        # print("hyper edge index shape:", hyper_edge_index.shape)
        # print("hyper edge attr shape:", hyper_edge_attr.shape)
        # print("hyper batch shape:", hyper_batch.shape)
        if hyper_edge_attr.dim() == 1:
            hyper_edge_attr = hyper_edge_attr.unsqueeze(-1)  
        in_channels = hyper_edge_attr.size(1) 
        if self.hgconv is None or self.hgconv.in_channels != in_channels:
            self.hgconv = HypergraphConv(in_channels, self.out_channels).to(hyper_edge_attr.device)

        # print("Hypergraph edge index before reindexing subjects:", hyper_edge_index)
        # Re-index hypergraph['edge_index'][0] to ensure uniqueness
        unique_subjects_hyper, inverse_indices_hyper = torch.unique(hyper_edge_index[0], return_inverse=True)
        hyper_edge_index[0] = inverse_indices_hyper  # Assign unique indices back to hypergraph['edge_index'][0]
        # print("Hypergraph edge index after reindexing subjects:", hyper_edge_index)
        # Re-index hypergraph['edge_index'][1] from 0 to hyper_edge_index.size()[1] - 1
        hyper_edge_index[1] = torch.arange(hyper_edge_index.size(1), device=hyper_edge_index.device)
        # print("Hypergraph edge index after reindexing nodes:", hyper_edge_index)
        # print("", hyper_edge_index.size())

        hyper_edge_features = self.hgconv(hyper_edge_attr, hyper_edge_index)  
        # print("Hypergraph edge features shape after HypergraphConv:", hyper_edge_features)

        unique_subjects, inverse_indices = torch.unique(hyper_batch, return_inverse=True)
        num_subjects = unique_subjects.size(0) 
        subject_features = torch.zeros((num_subjects, self.out_channels), device=hyper_edge_features.device)
        for i, subject_id in enumerate(unique_subjects):
            # Get indices for the current subject
            subject_indices = (hyper_batch == subject_id).nonzero(as_tuple=False).view(-1)

            # Use these indices to find all hyperedge IDs for the subject in hyper_edge_index[0]
            subject_hyper_edges = hyper_edge_index[0][subject_indices]

            # Remove duplicate hyperedge IDs
            unique_hyper_edges = torch.unique(subject_hyper_edges)

            # Use hyperedge IDs to find corresponding features in hyper_edge_features, and compute the mean
            subject_features[i] = hyper_edge_features[unique_hyper_edges].mean(dim=0)
        # print(subject_features)
        return subject_features

class EdgeAttrMLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, edge_attr):
        return self.mlp(edge_attr)

class SmriMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# Define the main graph classification model
class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_classes=2, heads=4,
                 dropout=0.15, num_prompts=3, moe_experts=4, moe_k=2, moe_thresh=0.8):
        super().__init__()
        effective_dropout = dropout
        # Positional encoding for graph nodes
        pe_dim = 8
        self.pe = LaplacianPE(k=pe_dim, model_dim=pe_dim)
        # MLP for edge attributes
        self.edge_mlp = EdgeAttrMLP(in_dim=1, out_dim=pe_dim)
        # MLP for sMRI features
        self.smri_mlp = SmriMLP(in_dim=2, out_dim=hidden_dim)
        # Scaling parameter for edge weights
        self.edge_weight_scale = nn.Parameter(torch.tensor(2.5))
        # Prompt embeddings for injecting additional nodes
        self.num_prompts = num_prompts
        self.prompt_dim = in_channels + pe_dim
        self.prompt_emb = nn.Parameter(torch.randn(num_prompts, self.prompt_dim))
        # GATConv layers for graph convolution
        self.conv1 = GATConv(in_channels + pe_dim + hidden_dim, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm2 = GraphNorm(hidden_dim * heads)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm3 = GraphNorm(hidden_dim * heads)
        # Jumping Knowledge layer for aggregating features
        self.jk = JumpingKnowledge(mode="max")
        # Global token for Transformer input
        self.global_token = nn.Parameter(torch.randn(1, hidden_dim * heads))
        # Transformer encoder for global context modeling
        self.trans_layer = TransformerEncoderLayer(
            d_model=hidden_dim * heads,  
            nhead=heads,
            dropout=effective_dropout
        )
        self.trans = TransformerEncoder(self.trans_layer, num_layers=2)
        self.trans_norm = nn.LayerNorm(hidden_dim * heads)  
        # Dropout layer
        self.dropout = nn.Dropout(p=effective_dropout)
        # MLP for subject features
        self.subject_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=effective_dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # MLP for sMRI features (redefined)
        self.smri_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=effective_dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Fusion layer for combining graph, subject, and sMRI features
        self.lin_fuse = nn.Sequential(
            nn.Linear(hidden_dim * heads + 2 * hidden_dim , hidden_dim), 
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        # Output layer for classification
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        # Scaling parameter for subject features
        self.subject_scale = nn.Parameter(torch.tensor(1.0))
        # Mixture of Experts (MoE) module
        self.moe = MoE(
            input_dim=hidden_dim,
            num_experts=moe_experts,
            k=moe_k, 
            thresh=moe_thresh, 
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        # Hypergraph capsule layer for feature transformation
        self.hyper_caps = HypergraphCapsule(hidden_dim)

    def forward(self, x, edge_index, edge_weight, batch, subject_feat, smri_feat, hypergraph):
        # Compute positional encoding for nodes
        pe = self.pe(edge_index=edge_index, batch=batch)
        x = torch.cat([x, pe], dim=1)
        # Process edge attributes using MLP
        edge_attr = self.edge_mlp(edge_weight.unsqueeze(-1))
        # Inject prompt nodes into the graph
        num_graphs = int(batch.max().item()) + 1
        # print("Number of graphs in batch:", num_graphs)
        prompt_feats = self.prompt_emb.unsqueeze(0).repeat(num_graphs, 1, 1)
        prompt_feats = prompt_feats.view(-1, self.prompt_dim)
        x_aug = torch.cat([x, prompt_feats], dim=0)
        prompt_batch = torch.arange(num_graphs, device=batch.device).unsqueeze(1)
        prompt_batch = prompt_batch.repeat(1, self.num_prompts).view(-1)
        batch_aug = torch.cat([batch, prompt_batch], dim=0)
        # Create edges between prompt nodes and original nodes
        num_orig_nodes = x.size(0)
        prompt_start = num_orig_nodes
        src_list, dst_list = [], []
        for i in range(num_graphs):
            nodes_i = (batch == i).nonzero(as_tuple=False).view(-1)
            p_idx = torch.arange(prompt_start + i * self.num_prompts,
                                 prompt_start + (i + 1) * self.num_prompts,
                                 device=batch.device)
            src_list.append(p_idx.unsqueeze(1).repeat(1, nodes_i.size(0)).flatten())
            dst_list.append(nodes_i.unsqueeze(0).repeat(self.num_prompts, 1).flatten())
            src_list.append(nodes_i.unsqueeze(0).repeat(self.num_prompts, 1).flatten())
            dst_list.append(p_idx.unsqueeze(1).repeat(1, nodes_i.size(0)).flatten())
        src_all = torch.cat(src_list)
        dst_all = torch.cat(dst_list)
        prompt_edges = torch.stack([src_all, dst_all], dim=0)
        edge_index_aug = torch.cat([edge_index, prompt_edges], dim=1)
        # Augment edge weights with zeros for prompt edges
        if edge_weight is not None:
            zero_w = torch.zeros(prompt_edges.size(1), device=edge_weight.device)
            edge_weight_aug = torch.cat([edge_weight, zero_w], dim=0)
        else:
            edge_weight_aug = None
        # Update graph with augmented nodes and edges
        x, edge_index, edge_weight, batch = x_aug, edge_index_aug, edge_weight_aug, batch_aug
        # print("Graph after prompt injection - num nodes:", x.size(0), "num edges:", edge_index.size(1))
        edge_attr = self.edge_mlp(edge_weight.unsqueeze(-1)) if edge_weight is not None else None

        # Hypergraph feature extraction
        hyper_edge_index = hypergraph['edge_index']
        hyper_edge_attr = hypergraph['edge_attr']
        hyper_edge_batch = hypergraph['batch']
        hyper_h = self.hyper_caps(hyper_edge_index, hyper_edge_attr, hyper_edge_batch)  # Extract hypergraph features
        # print("Hypergraph features shape after capsule:", hyper_h.shape)

        # Map hypergraph features to each node
        hyper_h_expanded = hyper_h[batch]  # Map each node to its corresponding subject features
        # print("Hypergraph features shape after expansion:", hyper_h_expanded.shape)

        # Concatenate hypergraph features to node features
        x = torch.cat([x, hyper_h_expanded], dim=1)
        # print("Node features shape after adding hypergraph features:", x.shape)

        # Apply GATConv layers with normalization and dropout
        h1 = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr) * self.edge_weight_scale))
        h1 = self.dropout(h1)
        h2 = F.elu(self.norm2(self.conv2(h1, edge_index, edge_attr) + h1))
        h2 = self.dropout(h2)
        h3 = F.elu(self.norm3(self.conv3(h2, edge_index, edge_attr) + h2))
        h3 = self.dropout(h3)
        # Aggregate features using Jumping Knowledge
        h_jk = self.jk([h1, h2, h3])

        # Convert graph features to dense batch representation
        h_dense, mask = to_dense_batch(h_jk, batch)
        B, N_max, D = h_dense.size()
        # Add global token for Transformer input
        g_token = self.global_token.unsqueeze(0).repeat(B, 1, 1)
        if g_token.size(2) != D:  # Ensure dimension consistency
            g_token = F.linear(g_token, torch.eye(D, g_token.size(2), device=g_token.device))
        h_dense = torch.cat([g_token, h_dense], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=mask.device, dtype=torch.bool), mask], dim=1)
        # Apply Transformer encoder for global context modeling
        x_trans_in = h_dense.permute(1, 0, 2)
        x_trans_out = self.trans(
            x_trans_in,
            src_key_padding_mask=~mask
        )
        x_trans_out = x_trans_out + x_trans_in
        h_trans = x_trans_out.permute(1, 0, 2)
        h_trans = self.trans_norm(h_trans)
        h_pool = h_trans[:, 0, :]  # Extract global token features
        # Process subject and sMRI features
        subject_emb = self.subject_mlp(subject_feat) * self.subject_scale
        smri_emb = self.smri_mlp(smri_feat)
        # Fuse graph, subject, and sMRI features
        fused = torch.cat([h_pool, subject_emb, smri_emb], dim=1)
        fused = self.lin_fuse(fused)
        fused = self.bn_fuse(fused)
        fused = F.elu(fused)
        fused = self.dropout(fused)
        # Apply Mixture of Experts (MoE) module
        fused = self.moe(fused)
        # Output classification logits
        return self.lin_out(fused)

# Define MoE module
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, k=None, hidden_dim=None, dropout=0.0, thresh=0.8):
        super().__init__()
        self.num_experts = num_experts
        self.k = sorted(k) if k else [1, num_experts]  
        self.thresh = thresh / num_experts 
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        hd = hidden_dim or input_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hd),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hd, input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_scores = self.gate(x) 

        mask = (gate_scores > self.thresh).float()  
        num_selected = mask.sum(dim=-1).long()  

        adjusted_k = []
        for n in num_selected:
            if n < self.k[0]:
                adjusted_k.append(self.k[0]) 
            elif n > self.k[-1]:
                adjusted_k.append(self.k[-1])  
            else:
                adjusted_k.append(min(self.k, key=lambda choice: abs(choice - n)))
        adjusted_k = torch.tensor(adjusted_k, device=x.device)  # [B]

        new_mask = torch.zeros_like(mask)
        for i, k in enumerate(adjusted_k):
            topk_vals, topk_idx = gate_scores[i].topk(k.item())  # Select top k experts
            new_mask[i, topk_idx] = 1.0

        gated = gate_scores * new_mask
        gated = gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, D]
        out = torch.einsum('be,bed->bd', gated, expert_outs)  # Weighted sum [B, D]
        return out

# Set random seed for reproducibility
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

# Main training and evaluation pipeline
def main():
    # Parse arguments and set random seed
    args = parse_args()
    set_seed(args.seed)

    # Load cached graph data
    args.graph_cache_dir = os.path.join(args.data_dir, 'graphs_cache')
    data_list = load_cached_graphs(args.graph_cache_dir)
    logger.info(f"Loaded graph samples from cache: {len(data_list)}")
    if len(data_list) == 0:
        logger.info("Cache is empty, exiting.")
        return

    # Attach subject features to graph data
    subj_features = load_subject_features(args.excel)
    for data in data_list:
        subj_id = extract_subject_id(getattr(data, "name", ""))
        if subj_id and subj_id in subj_features:
            data.subject_feat = torch.tensor(
                subj_features[subj_id], dtype=torch.float
            ).unsqueeze(0)
        else:
            logger.warning(f"Cannot find features for sample {getattr(data, 'name', 'unknown')}")
    logger.info("Attached subject features to graph data")
    
    # Split dataset into training, validation, and test sets
    total = len(data_list)
    test_n = max(1, int(total * args.test_size))
    val_n  = max(1, int(total * args.val_size))
    train_n= total - val_n - test_n
    lengths = [train_n, val_n, test_n]
    if sum(lengths) < total:
        lengths[0] += total - sum(lengths)
    train_ds, val_ds, test_ds = random_split(
        data_list, lengths, generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Count positive and negative samples in each dataset
    def count_labels(dataset):
        pos_count = sum(1 for data in dataset if data.y.item() == 1)
        neg_count = len(dataset) - pos_count
        return pos_count, neg_count

    train_pos, train_neg = count_labels(train_ds)
    val_pos, val_neg = count_labels(val_ds)
    test_pos, test_neg = count_labels(test_ds)

    # Log the sizes and label distributions of the datasets
    logger.info(f"Dataset sizes: Train={len(train_ds)}, Validation={len(val_ds)}, Test={len(test_ds)}")
    logger.info(f"Train set: Positive={train_pos}, Negative={train_neg}")
    logger.info(f"Validation set: Positive={val_pos}, Negative={val_neg}")
    logger.info(f"Test set: Positive={test_pos}, Negative={test_neg}")

    # Initialize model, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    args_moe_k_list = list(map(int, args.moe_k.split(',')))
    model = GraphClassifier(
        in_channels=1,
        hidden_dim=args.hidden_dim,
        heads=args.gat_heads,
        dropout=args.dropout,
        num_prompts=args.num_prompts,
        moe_experts=args.moe_experts,
        moe_k=args_moe_k_list,
        moe_thresh=args.moe_thresh
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.warmup_epochs, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.patience)

    # Define loss function
    class_weights = torch.tensor(list(map(float, args.class_weights.split(','))),
                                 dtype=torch.float,
                                 device=device)
    inv_weights   = 1.0 / class_weights
    criterion = FocalLoss(gamma=1.0,
                          weight=inv_weights,
                          smoothing=args.label_smoothing,
                          reduction='mean')

    # Training loop
    initial_lr = args.lr
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        if epoch <= args.warmup_epochs:
            warmup_lr = initial_lr * (epoch / args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        model.train()
        loss_sum, preds, trues, total_graphs = 0, [], [], 0
        for batch in train_loader:
            # Apply graph data augmentation, default is disabled
            batch = augment_graph(batch, drop_prob=-1, noise_std=-1, seed=42)
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                hypergraph=batch.hypergraph  # Pass hypergraph parameter
            )
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            preds += out.argmax(1).detach().cpu().tolist()
            trues += batch.y.cpu().tolist()
        avg_loss = loss_sum / total_graphs
        train_acc = accuracy_score(trues, preds)

        # Validation loop
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                    hypergraph=batch.hypergraph  # Pass hypergraph parameter
                )
                val_preds += out.argmax(1).cpu().tolist()
                val_trues += batch.y.cpu().tolist()
        val_acc = accuracy_score(val_trues, val_preds)

        # Log training and validation performance
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'origin_ADHD_622_best_graph_model_bf_cw_lf.pth')
            logger.info(f"Saved best model at epoch {epoch} with Val Acc: {val_acc:.4f}")

    # Test the best model
    model.load_state_dict(torch.load('origin_ADHD_622_best_graph_model_bf_cw_lf.pth'))
    test_preds, test_trues, test_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                hypergraph=batch.hypergraph  # Pass hypergraph parameter
            )
            probs = F.softmax(out, dim=1)[:, 1]
            test_probs += probs.cpu().tolist()
            test_preds += out.argmax(1).cpu().tolist()
            test_trues += batch.y.cpu().tolist()

    # Log test performance
    acc = accuracy_score(test_trues, test_preds)
    cm  = confusion_matrix(test_trues, test_preds)
    cr  = classification_report(
        test_trues,
        test_preds,
        target_names=['Normal','Disorder'],
        digits=4
    )
    logger.info("=== Test Set Classification Performance ===")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Confusion Matrix:\n" +
                np.array2string(
                    cm.astype(float),
                    formatter={'float_kind': lambda x: f"{x:.4f}"}
                ))
    logger.info("Classification Report:\n" + cr)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    f1  = f1_score(test_trues, test_preds)
    auc = roc_auc_score(test_trues, test_probs)
    logger.info(f"ACC (%): {acc*100:.2f}% | SEN (%): {sen*100:.2f}% | SPE (%): {spe*100:.2f}%")
    logger.info(f"F1-score: {f1:.4f} | AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
