import torch
import torch.nn as nn


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
        adjusted_k = torch.tensor(adjusted_k, device=x.device)

        new_mask = torch.zeros_like(mask)
        for i, k in enumerate(adjusted_k):
            topk_vals, topk_idx = gate_scores[i].topk(k.item())
            new_mask[i, topk_idx] = 1.0

        gated = gate_scores * new_mask
        gated = gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        out = torch.einsum('be,bed->bd', gated, expert_outs)
        return out