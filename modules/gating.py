import torch
import torch.nn as nn


class DualExpertGate(nn.Module):
    def __init__(self, lesion_dim, context_dim, hidden_dim=128, use_entropy=True):
        super().__init__()
        self.use_entropy = use_entropy
        in_dim = lesion_dim + context_dim + (1 if use_entropy else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lesion_feat, context_feat, entropy=None):
        if self.use_entropy:
            if entropy is None:
                raise ValueError("entropy is required when use_entropy=True")
            gate_in = torch.cat([lesion_feat, context_feat, entropy], dim=-1)
        else:
            gate_in = torch.cat([lesion_feat, context_feat], dim=-1)
        return torch.sigmoid(self.fc(gate_in))
