from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Dict

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (added to inputs).
    Works with batch_first tensors (B, T, D).
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerTabPFN(nn.Module):
    """
    Minimal Transformer encoder that reads packed tokens and predicts labels/values at query positions.
    Input: tokens (B, T, D_in) where last dims include features + (label onehot or scalar) + known_flag.
    Output:
      - classification: logits (B, T, C)
      - regression:     values (B, T)
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        task_type: str = "cls",
        num_classes: Optional[int] = None,
        max_len: int = 1024,
    ):
        super().__init__()
        assert task_type in {"cls", "reg"}
        if task_type == "cls":
            assert num_classes is not None and num_classes >= 2, "num_classes required for classification"

        self.task_type = task_type
        self.num_classes = num_classes

        self.inp = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if task_type == "cls":
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens:    (B, T, D_in)
        attn_mask: (B, T) True for valid tokens (non-pad)
        """
        h = self.inp(tokens)
        h = self.pos(h)

        # MPS (Apple GPU) workaround: skip key_padding_mask to avoid unsupported op.
        if tokens.device.type == "mps":
            # Zero padded positions so they contribute minimally.
            h = h * attn_mask.unsqueeze(-1).to(h.dtype)
            key_padding_mask = None
        else:
            # PyTorch expects True where positions are PAD (to mask). We have True for VALID → invert.
            key_padding_mask = ~attn_mask  # (B, T) True = pad

        h = self.enc(h, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)
        out = self.head(h)  # (B, T, C) or (B, T, 1)
        if self.task_type == "reg":
            out = out.squeeze(-1)  # (B, T)
        return {"out": out}
