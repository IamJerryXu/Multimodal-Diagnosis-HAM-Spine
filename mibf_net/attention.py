import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        flattened = x.view(batch_size, channels, -1).permute(0, 2, 1)
        query = self.query(flattened)
        key = self.key(flattened).permute(0, 2, 1)
        value = self.value(flattened)
        attention_scores = torch.bmm(query, key)
        attention_scores = self.softmax(attention_scores / (channels ** 0.5))
        attended_values = torch.bmm(attention_scores, value)
        return attended_values.permute(0, 2, 1).view(batch_size, channels, height, width)


def compute_kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, min=eps, max=1.0)
    q = torch.clamp(q, min=eps, max=1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


class MultiHeadCrossAttention_v2(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads")

        self.toK_x = nn.Linear(dim, dim)
        self.toQ_x = nn.Linear(dim, dim)
        self.toV_x = nn.Linear(dim, dim)
        self.toK_y = nn.Linear(dim, dim)
        self.toV_y = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, y):
        batch_size, seq_len_x, _ = x.shape
        _, seq_len_y, _ = y.shape

        Kx = self.toK_x(x)
        Qx = self.toQ_x(x)
        Vx = self.toV_x(x)
        Ky = self.toK_y(y)
        Vy = self.toV_y(y)

        Kx = Kx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Qx = Qx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Vx = Vx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Ky = Ky.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)
        Vy = Vy.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)

        Kcat = torch.cat([Kx, Ky], dim=2)
        Vcat = torch.cat([Vx, Vy], dim=2)
        attention_scores = torch.matmul(Qx, Kcat.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, Vcat)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_x, self.dim)
        return self.to_out(output)
