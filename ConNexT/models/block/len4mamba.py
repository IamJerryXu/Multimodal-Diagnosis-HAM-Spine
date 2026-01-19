#（16，768）是经过BERT编码器的首token向量；第二个向量是[16, 640, 7, 7]->（16，640，49）；第三个向量和第四个向量是(8, 720, 3584) (8, 720, 3584)，怎么拼接这些向量通过mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # 假设使用标准Mamba实现
from .kan1 import KAN1 


    #     text: [B, 768]
    #     img: [B, 640, 49] → 应先转为 [B, 49, 640]
    #     first_hidden, last_hidden: [B, 3584]

    #     # [B, 768] -> [B, 1, 256]
    #     # [B, 640, 49] -> [B, 49, 640] -> [B, 49, 256]
    #     # [B, 3584] -> [B, 1, 256]

    #     # 拼接序列: [B, 52, 256]
    #     return self.mamba(concat_seq)  # [B, seq_len, 256]


class KANMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.head_dim = embed_dim // num_heads

        # 使用 KAN 替代传统 Linear
        self.q_proj = KAN1([embed_dim, embed_dim])
        self.k_proj = KAN1([embed_dim, embed_dim])
        self.v_proj = KAN1([embed_dim, embed_dim])

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        # 计算 QKV: [B, L, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 分成 num_heads 个头: [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score: [B, num_heads, L, L]
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = attn_weights @ v  # [B, num_heads, L, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(attn_output)  # [B, L, D]
    
class MultimodalMambaWithKANAttention(nn.Module):
    def __init__(self, text_dim=768, img_dim=640, hidden_dim=3584, proj_dim=256, num_heads=4):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, proj_dim)
        self.proj_img = nn.Linear(img_dim, proj_dim)
        self.proj_first = nn.Linear(hidden_dim, proj_dim)
        self.proj_last = nn.Linear(hidden_dim, proj_dim)

        self.attn = KANMultiheadAttention(embed_dim=proj_dim, num_heads=num_heads)

        self.mamba = Mamba(
            d_model=proj_dim,
            d_state=128,
            d_conv=4,
            expand=2
        )

        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)

        self.positional_encoding = self._create_positional_encoding(max_len=2048, d_model=proj_dim)

    def forward(self, text, img, first_hidden, last_hidden):
        B = text.size(0)

        # 文本特征: [B, 768] -> [B, 1, 256]
        text_proj = self.proj_text(text).unsqueeze(1)

        # 图像特征: [B, 640, 7, 7] → [B, 49, 640] → [B, 49, 256]
        img = img.permute(0, 2, 1)
        img_proj = self.proj_img(img)

        # 首尾特征: [B, 3584] -> [B, 1, 256]
        first_proj = self.proj_first(first_hidden).unsqueeze(1)
        last_proj = self.proj_last(last_hidden).unsqueeze(1)

        # 拼接序列: [B, 52, 256]
        concat_seq = torch.cat([text_proj, img_proj, first_proj, last_proj], dim=1)

        # === 添加位置编码 ===
        if self.positional_encoding is not None:
            pe = self.positional_encoding[:, :concat_seq.size(1), :].to(concat_seq.device)
            concat_seq = concat_seq + pe

        # === Step 1: KAN 注意力 ===
        attn_output = self.attn(concat_seq)
        attn_output = self.norm1(attn_output + concat_seq)

        # === Step 2: Mamba Block ===
        mamba_output = self.mamba(attn_output)
        output = self.norm2(mamba_output + attn_output)

        return output  # [B, 52, 256]
    def _create_positional_encoding(self, max_len=1024, d_model=256):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe





#使用 Mamba 模块处理多模态输入，但是没有使用注意力机制
class MultimodalMamba(nn.Module):
    def __init__(self, text_dim=768, img_dim=1568, hidden_dim=3584, proj_dim=256):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, proj_dim)
        self.proj_img = nn.Linear(img_dim, proj_dim)
        self.proj_first = nn.Linear(hidden_dim, proj_dim)
        self.proj_last = nn.Linear(hidden_dim, proj_dim)

        self.mamba = Mamba(
            d_model=proj_dim,
            d_state=128,
            d_conv=4,
            expand=2
        )

        self.positional_encoding = self._create_positional_encoding(max_len=2048, d_model=proj_dim)

    def forward(self, text, img, first_hidden, last_hidden):

        B = text.size(0)

        # [B, 768] -> [B, 1, 256]
        text_proj = self.proj_text(text).unsqueeze(1)

        # [B, 640, 49] -> [B, 49, 640] -> [B, 49, 256]
        img = img.permute(0, 2, 1)
        img_proj = self.proj_img(img)

        # [B, 3584] -> [B, 1, 256]
        first_proj = self.proj_first(first_hidden).unsqueeze(1)
        last_proj = self.proj_last(last_hidden).unsqueeze(1)

        # 拼接序列: [B, 52, 256]
        concat_seq = torch.cat([text_proj, img_proj, first_proj, last_proj], dim=1)

        # 添加位置编码
        if self.positional_encoding is not None:
            pe = self.positional_encoding[:, :concat_seq.size(1), :].to(concat_seq.device)
            concat_seq = concat_seq + pe

        # ===== 残差连接 =====
        residual = concat_seq
        mamba_out = self.mamba(concat_seq)

        output = mamba_out + residual

        return output  # [B, seq_len, 256]


    def _create_positional_encoding(self, max_len=1024, d_model=256):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
