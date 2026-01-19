import os
import sys
import torch
import torch.nn as nn


class BasicTransformerBlock(nn.Module):
    """
    Transformer block inspired by Stable Diffusion's BasicTransformerBlock.
    Contains:
    1. Self-Attention (Image attending to Image)
    2. Cross-Attention (Image attending to Text)
    3. FeedForward
    """

    def __init__(self, dim, context_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        # Cross-attention: Query is dim, Key/Value is context_dim
        self.attn2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, context, context_mask=None):
        # x: (B, N, dim) - Image tokens
        # context: (B, Nt, context_dim) - Text tokens

        # 1. Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn1(x, x, x)
        x = residual + x

        # 2. Cross-Attention
        residual = x
        x = self.norm2(x)

        # Handle mask: context_mask is 1 for valid, 0 for padding.
        # MHA expects True for ignored positions.
        kp_mask = None
        if context_mask is not None:
            kp_mask = context_mask == 0

        x, _ = self.attn2(x, context, context, key_padding_mask=kp_mask)
        x = residual + x

        # 3. FeedForward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x

        return x


class FusionModule(nn.Module):
    """
    Fusion module using BasicTransformerBlock.
    Image tokens are the primary stream, enriched by Text context.
    """

    def __init__(self, text_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.transformer_block = BasicTransformerBlock(
            hidden_dim, text_dim, num_heads, dropout
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img_tokens, txt_tokens, txt_mask=None):
        """
        img_tokens: (B, N, H)
        txt_tokens: (B, Nt, text_dim)
        txt_mask: (B, Nt) optional
        """
        # Apply Transformer Block
        x = self.transformer_block(img_tokens, txt_tokens, txt_mask)

        # Global Average Pooling: (B, N, H) -> (B, H)
        x = x.transpose(1, 2)  # (B, H, N)
        x = self.pool(x).squeeze(2)  # (B, H)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.txt_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, img_tokens, txt_tokens, txt_mask=None):
        """
        img_tokens: (B, N, hidden_dim)
        txt_tokens: (B, Nt, text_dim)
        """
        txt_proj = self.txt_proj(txt_tokens)
        key_padding_mask = None
        if txt_mask is not None:
            key_padding_mask = txt_mask == 0

        attn_out, _ = self.attn(
            img_tokens, txt_proj, txt_proj, key_padding_mask=key_padding_mask
        )
        return self.norm(img_tokens + attn_out)


class MultiScaleFusionModule(nn.Module):
    """
    Multi-scale cross-attention fusion for layer2/3/4 image tokens.
    """

    def __init__(self, text_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_l2 = CrossAttentionBlock(text_dim, hidden_dim, num_heads, dropout)
        self.cross_l3 = CrossAttentionBlock(text_dim, hidden_dim, num_heads, dropout)
        self.cross_l4 = CrossAttentionBlock(text_dim, hidden_dim, num_heads, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _pool_tokens(self, tokens):
        tokens = tokens.transpose(1, 2)
        return self.pool(tokens).squeeze(-1)

    def forward(self, img_tokens, txt_tokens, txt_mask=None):
        """
        img_tokens: dict with keys layer2/3/4
        txt_tokens: (B, Nt, text_dim)
        """
        t2 = self.cross_l2(img_tokens["layer2"], txt_tokens, txt_mask)
        t3 = self.cross_l3(img_tokens["layer3"], txt_tokens, txt_mask)
        t4 = self.cross_l4(img_tokens["layer4"], txt_tokens, txt_mask)

        p2 = self._pool_tokens(t2)
        p3 = self._pool_tokens(t3)
        p4 = self._pool_tokens(t4)

        return (p2 + p3 + p4) / 3.0


class ConcatFusionModule(nn.Module):
    def __init__(self, text_dim, hidden_dim, text_pool="cls"):
        super().__init__()
        self.text_pool = text_pool
        self.proj = nn.Linear(hidden_dim + text_dim, hidden_dim)

    def _pool_text(self, text_tokens):
        if self.text_pool == "mean":
            return text_tokens.mean(dim=1)
        return text_tokens[:, 0, :]

    def _pool_image(self, image_tokens):
        if isinstance(image_tokens, dict):
            pooled = []
            for key in ("layer2", "layer3", "layer4"):
                tokens = image_tokens[key]
                pooled.append(tokens.mean(dim=1))
            return sum(pooled) / float(len(pooled))
        return image_tokens.mean(dim=1)

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        img_feat = self._pool_image(image_tokens)
        txt_feat = self._pool_text(text_tokens)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.proj(fused)


class WeightedConcatFusionModule(ConcatFusionModule):
    def __init__(self, text_dim, hidden_dim, text_pool="cls"):
        super().__init__(text_dim, hidden_dim, text_pool=text_pool)
        self.w_img = nn.Parameter(torch.zeros(1))
        self.w_txt = nn.Parameter(torch.zeros(1))

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        img_feat = self._pool_image(image_tokens)
        txt_feat = self._pool_text(text_tokens)
        w_img = torch.sigmoid(self.w_img)
        w_txt = torch.sigmoid(self.w_txt)
        fused = torch.cat([img_feat * w_img, txt_feat * w_txt], dim=1)
        return self.proj(fused)


class HadamardFusionModule(nn.Module):
    def __init__(self, text_dim, hidden_dim, text_pool="cls"):
        super().__init__()
        self.text_pool = text_pool
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)
        self.txt_proj = nn.Linear(text_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _pool_text(self, text_tokens):
        if self.text_pool == "mean":
            return text_tokens.mean(dim=1)
        return text_tokens[:, 0, :]

    def _pool_image(self, image_tokens):
        if isinstance(image_tokens, dict):
            pooled = []
            for key in ("layer2", "layer3", "layer4"):
                tokens = image_tokens[key]
                pooled.append(tokens.mean(dim=1))
            return sum(pooled) / float(len(pooled))
        return image_tokens.mean(dim=1)

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        img_feat = self._pool_image(image_tokens)
        txt_feat = self._pool_text(text_tokens)
        fused = self.img_proj(img_feat) * self.txt_proj(txt_feat)
        return self.norm(fused)


class BilinearFusionModule(nn.Module):
    def __init__(self, text_dim, hidden_dim, text_pool="cls", rank=128):
        super().__init__()
        self.text_pool = text_pool
        self.img_proj = nn.Linear(hidden_dim, rank)
        self.txt_proj = nn.Linear(text_dim, rank)
        self.out_proj = nn.Linear(rank, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _pool_text(self, text_tokens):
        if self.text_pool == "mean":
            return text_tokens.mean(dim=1)
        return text_tokens[:, 0, :]

    def _pool_image(self, image_tokens):
        if isinstance(image_tokens, dict):
            pooled = []
            for key in ("layer2", "layer3", "layer4"):
                tokens = image_tokens[key]
                pooled.append(tokens.mean(dim=1))
            return sum(pooled) / float(len(pooled))
        return image_tokens.mean(dim=1)

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        img_feat = self._pool_image(image_tokens)
        txt_feat = self._pool_text(text_tokens)
        fused = self.img_proj(img_feat) * self.txt_proj(txt_feat)
        return self.norm(self.out_proj(fused))


class SSMFusionModule(nn.Module):
    def __init__(self, text_dim, hidden_dim, text_pool="cls"):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError(
                "SSM/Mamba fusion requires `mamba-ssm`. Install with `pip install mamba-ssm`."
            ) from exc

        self.text_pool = text_pool
        self.txt_proj = nn.Linear(text_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _pool_text(self, text_tokens):
        if self.text_pool == "mean":
            return text_tokens.mean(dim=1)
        return text_tokens[:, 0, :]

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        # image_tokens: (B, N, H)
        if isinstance(image_tokens, dict):
            raise ValueError("SSMFusionModule expects single-scale image tokens.")
        txt_feat = self.txt_proj(self._pool_text(text_tokens)).unsqueeze(1)
        tokens = image_tokens + txt_feat
        tokens = self.mamba(tokens)
        tokens = tokens.transpose(1, 2)
        return self.pool(tokens).squeeze(2)


class VMambaFusionModule(nn.Module):
    def __init__(self, text_dim, hidden_dim, text_pool="cls", vmamba_dim=32):
        super().__init__()
        try:
            from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block
        except Exception:
            vmamba_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "EnergeSnake")
            )
            sys.path.insert(0, vmamba_root)
            try:
                from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block
            except Exception as exc:
                raise ImportError(
                    "VMamba not found. Ensure EnergeSnake is present at /data/QLI/EnergeSnake."
                ) from exc

        self.text_pool = text_pool
        self.vmamba_dim = vmamba_dim
        self.in_proj = nn.Linear(hidden_dim, vmamba_dim)
        self.txt_proj = nn.Linear(text_dim, vmamba_dim)
        num_heads = max(1, vmamba_dim // 16)
        self.vmamba = VMAMBA2Block(dim=vmamba_dim, num_heads=num_heads)
        self.out_proj = nn.Linear(vmamba_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _pool_text(self, text_tokens):
        if self.text_pool == "mean":
            return text_tokens.mean(dim=1)
        return text_tokens[:, 0, :]

    def forward(self, image_tokens, text_tokens, txt_mask=None):
        if isinstance(image_tokens, dict):
            raise ValueError("VMambaFusionModule expects single-scale image tokens.")
        txt_feat = self.txt_proj(self._pool_text(text_tokens)).unsqueeze(1)
        tokens = self.in_proj(image_tokens) + txt_feat
        tokens, _ = self.vmamba(tokens, H=tokens.size(1), W=1)
        tokens = self.out_proj(tokens)
        tokens = tokens.transpose(1, 2)
        return self.pool(tokens).squeeze(2)
__all__ = [
    "FusionModule",
    "ConcatFusionModule",
    "MultiScaleFusionModule",
    "WeightedConcatFusionModule",
    "HadamardFusionModule",
    "BilinearFusionModule",
    "SSMFusionModule",
    "VMambaFusionModule",
]
