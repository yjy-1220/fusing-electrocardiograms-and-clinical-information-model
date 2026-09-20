"""Single-modal ECG model definition (M_ECG): encoder + classification head / SimCLR pretraining head.

The ECGWav2VecEncoder is shared by both training scripts; see model.common.ecg_encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.ecg_encoder import ECGWav2VecEncoder


class ClassificationHead(nn.Module):
    """5-layer MLP (Linear + GELU + Dropout 0.2, width kept at 512)
    + binary linear classifier."""

    def __init__(self, dim: int = 512, num_layers: int = 5, dropout: float = 0.2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        return self.mlp(h)  # (B, 1) logits


class ECGClassifier(nn.Module):
    """Single-modal ECG model M_ECG: encoder + classification head, outputs OMI logits."""

    def __init__(self, num_leads: int, hidden_dim: int, num_layers: int,
                 mlp_layers: int, dropout: float):
        super().__init__()
        self.encoder = ECGWav2VecEncoder(num_leads, hidden_dim, num_layers)
        self.head = ClassificationHead(hidden_dim, mlp_layers, dropout)

    def forward(self, x):
        return self.head(self.encoder(x))


class ECGSimCLR(nn.Module):
    """Pretraining model: encoder + SimCLR projection head.
    The projection head is discarded during fine-tuning."""

    def __init__(self, encoder: ECGWav2VecEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.norm.normalized_shape[0], encoder.norm.normalized_shape[0]),
            nn.ReLU(inplace=True),
            nn.Linear(encoder.norm.normalized_shape[0], proj_dim),
        )

    def forward(self, x):
        return self.projection(self.encoder(x))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent contrastive loss. z1/z2: (B, D)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)                      # (2B, D)
    sim = torch.mm(z, z.t()) / temperature              # (2B, 2B)
    b = z1.size(0)
    mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)], dim=0).to(z.device)
    return F.cross_entropy(sim, targets)
