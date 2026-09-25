import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv


class HypergraphCapsule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.hgconv = None

    def forward(self, hyper_edge_index, hyper_edge_attr, hyper_batch):
        if hyper_edge_attr.dim() == 1:
            hyper_edge_attr = hyper_edge_attr.unsqueeze(-1)
        in_channels = hyper_edge_attr.size(1)
        if self.hgconv is None or self.hgconv.in_channels != in_channels:
            self.hgconv = HypergraphConv(in_channels, self.out_channels).to(hyper_edge_attr.device)

        # Re-index hyper_edge_index[0] to ensure uniqueness across subjects
        unique_subjects_hyper, inverse_indices_hyper = torch.unique(hyper_edge_index[0], return_inverse=True)
        hyper_edge_index[0] = inverse_indices_hyper

        # Re-index hyper_edge_index[1] from 0 to N-1
        hyper_edge_index[1] = torch.arange(hyper_edge_index.size(1), device=hyper_edge_index.device)

        hyper_edge_features = self.hgconv(hyper_edge_attr, hyper_edge_index)

        # Aggregate hyper-edge features per subject
        unique_subjects, inverse_indices = torch.unique(hyper_batch, return_inverse=True)
        num_subjects = unique_subjects.size(0)
        subject_features = torch.zeros((num_subjects, self.out_channels), device=hyper_edge_features.device)
        for i, subject_id in enumerate(unique_subjects):
            subject_indices = (hyper_batch == subject_id).nonzero(as_tuple=False).view(-1)
            subject_hyper_edges = hyper_edge_index[0][subject_indices]
            unique_hyper_edges = torch.unique(subject_hyper_edges)
            subject_features[i] = hyper_edge_features[unique_hyper_edges].mean(dim=0)
        return subject_features