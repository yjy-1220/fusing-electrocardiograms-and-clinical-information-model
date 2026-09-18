"""ECG Wav2Vec encoder: Conv stack + Transformer + mean pooling.


"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvFeatureExtractor(nn.Module):
    """
    Front-end 5-layer 1D convolutional stack: extracts local waveform features.
    Input (B, num_leads, L) -> output (B, hidden_dim, L').
    Downsampled sequence lengths: 250 -> 125 -> 63 -> 32 -> 32 -> 32.
    """

    def __init__(self, num_leads: int, hidden_dim: int):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.convs = nn.Sequential(
            ConvBlock(num_leads, c1, kernel=7, stride=2, pad=3),
            ConvBlock(c1, c2, kernel=5, stride=2, pad=2),
            ConvBlock(c2, c3, kernel=3, stride=2, pad=1),
            ConvBlock(c3, hidden_dim, kernel=3, stride=1, pad=1),
            ConvBlock(hidden_dim, hidden_dim, kernel=3, stride=1, pad=1),
        )

    def forward(self, x):
        return self.convs(x)


class ECGWav2VecEncoder(nn.Module):
    """Custom Wav2Vec-style ECG encoder: Conv stack + 12-layer Transformer
    + average pooling -> 512-dim embedding."""

    def __init__(self, num_leads: int = 12, hidden_dim: int = 512,
                 num_layers: int = 12, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = ConvFeatureExtractor(num_leads, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, leads, L)
        h = self.conv_stack(x)          # (B, D, L')
        h = h.transpose(1, 2)           # (B, L', D)
        h = self.transformer(h)         # (B, L', D)
        h = h.mean(dim=1)               # average pooling -> (B, D)
        return self.norm(h)
