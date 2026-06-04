# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---------- OHLCV Sequence Transformer ----------
class OHLCVTransformer(nn.Module):
    """Temporal transformer for OHLCV sequences."""
    def __init__(self, input_dim=4, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, DIM_FEEDFORWARD, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)   # global average pooling over time

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)   # (batch, d_model)
        return x

# ---------- Vision Transformer (ViT) ----------
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3, embed_dim=D_MODEL):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # (B, E, H/P, W/P)
        x = x.flatten(2)          # (B, E, N)
        x = x.transpose(1,2)      # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000, d_model=D_MODEL, depth=NUM_LAYERS, heads=NHEAD, dim_head=64, dropout=DROPOUT):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, DIM_FEEDFORWARD, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.pool = nn.Identity()   # we'll use cls token

    def forward(self, img):
        # img: (B, C, H, W)
        x = self.patch_embed(img)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        cls_out = x[:, 0]   # (B, d_model)
        return cls_out

# ---------- Multi-Timeframe Fusion ----------
class TimeframeFusion(nn.Module):
    """Attention-based fusion of multiple timeframe representations."""
    def __init__(self, d_model=D_MODEL, num_timeframes=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tf_features):
        # tf_features: list of (B, d_model) tensors, one per timeframe
        stacked = torch.stack(tf_features, dim=1)  # (B, T, d_model)
        attn_out, _ = self.attn(stacked, stacked, stacked)
        fused = attn_out.mean(dim=1)   # simple mean, or weighted via learnable
        fused = self.norm(fused + stacked.mean(dim=1))
        return fused