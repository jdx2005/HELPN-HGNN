import torch
import torch.nn as nn

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