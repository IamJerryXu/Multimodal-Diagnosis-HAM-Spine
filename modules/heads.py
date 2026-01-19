import os
import sys
import torch
import torch.nn as nn


def _load_groupkan():
    try:
        from ikan.GroupKAN import GroupKANLinear
        return GroupKANLinear
    except Exception:
        _ikan_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "Efficient-KAN-in-Chinese")
        )
        if os.path.isdir(os.path.join(_ikan_root, "ikan")):
            sys.path.insert(0, _ikan_root)
            try:
                from ikan.GroupKAN import GroupKANLinear
                return GroupKANLinear
            except Exception:
                return None
        return None


GroupKANLinear = _load_groupkan()


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return self.norm(residual + out)


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.res_block = ResidualBlock(hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.project(x)
        x = self.act(x)
        x = self.res_block(x)
        return self.classifier(x)


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        # input_dim is hidden_dim from fusion (single scale)
        self.hidden_dim = hidden_dim
        # We don't need to reshape anymore if input is already (B, H)
        # But wait, AttentionPooling usually pools over a sequence.
        # If the input is already pooled (B, H), then AttentionPooling is just a linear layer or self-attention?
        # The previous implementation assumed input was (B, 3*H) and reshaped to (B, 3, H).
        # Now input is (B, H).
        # If we want to keep "AttentionPooling", maybe we should apply it BEFORE pooling in CrossAttentionBlock?
        # But CrossAttentionBlock already pools.
        # So "AttentionPoolingClassifier" on top of a pooled vector (B, H) doesn't make much sense unless we project it to something else.
        # Let's just make it a Self-Attention on the single vector? No, that's trivial.
        # Let's assume if the user selects "attention_pooling" with single scale,
        # it might just be a simple projection or we can remove it.
        # However, to keep the code running, let's just make it a simple MLP with a skip connection or similar,
        # OR, we can interpret "AttentionPooling" as:
        # The fusion output is (B, H).
        # We can just use a Linear layer.
        # BUT, if we want to support the "AttentionPooling" option in config, we should adapt it.
        # Let's make it a simple projection -> attention -> classifier?
        # Actually, if input is (B, H), we can treat it as sequence length 1.

        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, H)
        B = x.size(0)

        # Treat as sequence of length 1: (B, 1, H)
        x = x.unsqueeze(1)

        # Query: (B, 1, H)
        q = self.query.expand(B, -1, -1)

        # Attention
        attn_out, _ = self.attn(q, x, x)  # (B, 1, H)

        out = attn_out.squeeze(1)  # (B, H)
        return self.classifier(out)


def build_kan_head(
    hidden_dim,
    num_classes,
    dropout=0.1,
    num_groups=8,
    act_mode="gelu",
):
    if GroupKANLinear is None:
        raise ImportError(
            "GroupKANLinear not found. Install the ikan package to use classifier_type='kan'."
        )
    if hidden_dim % num_groups != 0:
        raise ValueError(
            f"kan_num_groups ({num_groups}) must divide hidden_dim ({hidden_dim})."
        )

    return nn.Sequential(
        GroupKANLinear(
            hidden_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
        ),
        nn.LayerNorm(hidden_dim),
        GroupKANLinear(
            hidden_dim,
            num_classes,
            act_mode=act_mode,
            drop=0.0,
            num_groups=num_groups,
        ),
    )


__all__ = [
    "ResidualClassifier",
    "AttentionPoolingClassifier",
    "build_kan_head",
]
