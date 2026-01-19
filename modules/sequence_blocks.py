import math
import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        encoder_type="lstm",
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
        num_heads=4,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.hidden_dim = hidden_dim

        if self.encoder_type in ("lstm", "gru"):
            rnn_cls = nn.LSTM if self.encoder_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            self.proj = nn.Linear(output_dim, hidden_dim) if output_dim != hidden_dim else nn.Identity()
        elif self.encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=max(hidden_dim * 4, input_dim * 2),
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        else:
            raise ValueError(f"Unsupported sequence encoder type: {encoder_type}")

    def _positional_encoding(self, seq_len, dim, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (B, T, D)
        if self.encoder_type in ("lstm", "gru"):
            out, _ = self.rnn(x)
            last = out[:, -1, :]
            return self.proj(last)

        seq_len = x.size(1)
        pos = self._positional_encoding(seq_len, x.size(-1), x.device)
        x = x + pos.unsqueeze(0)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        return self.proj(pooled)
