import torch.nn as nn


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