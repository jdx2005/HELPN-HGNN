import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATConv, JumpingKnowledge, GraphNorm
from torch_geometric.utils import to_dense_batch

from .edge_attr_mlp import EdgeAttrMLP
from .smri_mlp import SmriMLP
from .moe import MoE
from .hypergraph_capsule import HypergraphCapsule
from ..utils.laplacian_pe import LaplacianPE


class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_classes=2, heads=4,
                 dropout=0.15, num_prompts=3, moe_experts=4, moe_k=2, moe_thresh=0.8):
        super().__init__()
        effective_dropout = dropout

        pe_dim = 8
        self.pe = LaplacianPE(k=pe_dim, model_dim=pe_dim)
        self.edge_mlp = EdgeAttrMLP(in_dim=1, out_dim=pe_dim)
        self.smri_mlp = SmriMLP(in_dim=2, out_dim=hidden_dim)
        self.edge_weight_scale = nn.Parameter(torch.tensor(2.5))

        self.num_prompts = num_prompts
        self.prompt_dim = in_channels + pe_dim
        self.prompt_emb = nn.Parameter(torch.randn(num_prompts, self.prompt_dim))

        self.conv1 = GATConv(in_channels + pe_dim + hidden_dim, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm2 = GraphNorm(hidden_dim * heads)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                             dropout=effective_dropout, edge_dim=pe_dim)
        self.norm3 = GraphNorm(hidden_dim * heads)

        self.jk = JumpingKnowledge(mode="max")

        self.global_token = nn.Parameter(torch.randn(1, hidden_dim * heads))
        self.trans_layer = TransformerEncoderLayer(
            d_model=hidden_dim * heads,
            nhead=heads,
            dropout=effective_dropout
        )
        self.trans = TransformerEncoder(self.trans_layer, num_layers=2)
        self.trans_norm = nn.LayerNorm(hidden_dim * heads)

        self.dropout = nn.Dropout(p=effective_dropout)

        self.subject_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=effective_dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.smri_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=effective_dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lin_fuse = nn.Sequential(
            nn.Linear(hidden_dim * heads + 2 * hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)

        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.subject_scale = nn.Parameter(torch.tensor(1.0))

        self.moe = MoE(
            input_dim=hidden_dim,
            num_experts=moe_experts,
            k=moe_k,
            thresh=moe_thresh,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.hyper_caps = HypergraphCapsule(hidden_dim)

    def forward(self, x, edge_index, edge_weight, batch, subject_feat, smri_feat, hypergraph):
        # Positional encoding for graph nodes
        pe = self.pe(edge_index=edge_index, batch=batch)
        x = torch.cat([x, pe], dim=1)

        # Process edge attributes
        edge_attr = self.edge_mlp(edge_weight.unsqueeze(-1))

        # Inject prompt nodes into the graph
        num_graphs = int(batch.max().item()) + 1
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
        edge_attr = self.edge_mlp(edge_weight.unsqueeze(-1)) if edge_weight is not None else None

        # Hypergraph feature extraction
        hyper_edge_index = hypergraph['edge_index']
        hyper_edge_attr = hypergraph['edge_attr']
        hyper_edge_batch = hypergraph['batch']
        hyper_h = self.hyper_caps(hyper_edge_index, hyper_edge_attr, hyper_edge_batch)

        # Map hypergraph features to each node (per-subject)
        hyper_h_expanded = hyper_h[batch]

        # Concatenate hypergraph features to node features
        x = torch.cat([x, hyper_h_expanded], dim=1)

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
        if g_token.size(2) != D:
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
        h_pool = h_trans[:, 0, :]

        # Process subject and sMRI features
        subject_emb = self.subject_mlp(subject_feat) * self.subject_scale
        smri_emb = self.smri_mlp(smri_feat)

        # Fuse graph, subject, and sMRI features
        fused = torch.cat([h_pool, subject_emb, smri_emb], dim=1)
        fused = self.lin_fuse(fused)
        fused = self.bn_fuse(fused)
        fused = F.elu(fused)
        fused = self.dropout(fused)

        # Apply Mixture of Experts module
        fused = self.moe(fused)

        return self.lin_out(fused)