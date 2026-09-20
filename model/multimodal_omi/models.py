"""Multimodal fusion model definition: ECG encoder + MacBERT text branch + Transformer fusion.

The ECG encoder ECGWav2VecEncoder is shared by both training scripts; see model.common.ecg_encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common.ecg_encoder import ECGWav2VecEncoder


class ECGMlpProjection(nn.Module):
    """Paper: 5-layer MLP (512->512, GELU, dropout 0.2) + separate linear
    projection 512 -> 256."""

    def __init__(self, dim: int = 512, num_layers: int = 5, dropout: float = 0.2,
                 out_dim: int = 256):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, h):
        return self.proj(self.mlp(h))


class GeGLUFFN(nn.Module):
    """Gated Gaussian-error-linear-unit feed-forward network.
    expansion: 256 -> 512 -> 256."""

    def __init__(self, dim: int = 256, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.gate(x)) * self.up(x)))


class FusionTransformerLayer(nn.Module):
    """Paper fusion layer: multi-head self-attention + LayerNorm +
    GeGLU FFN (256->512->256). Pre-LN style, dropout 0.1."""

    def __init__(self, dim: int = 256, nhead: int = 8, dropout: float = 0.1,
                 ffn_expansion: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GeGLUFFN(dim, ffn_expansion, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MultimodalOMIClassifier(nn.Module):
    """
    Full multimodal model M_multi:
      ECG   : Wav2Vec encoder -> 5-layer MLP -> Linear(512->256)
      Text  : MacBERT (CLS, 768) -> Linear(768->256)
      Fusion: [fusion-CLS, ECG-token, text-token] -> 5x FusionTransformerLayer
              -> fusion-CLS -> Linear(256->1)

    mode='text'  -> Mclin (MacBERT CLS -> Linear(768->256) -> head)
    mode='ecg'   -> MECG  (Wav2Vec -> MLP -> Linear(512->256) -> head)
    """

    def __init__(self, macbert_path: str, mode: str = "fusion",
                 num_leads: int = 12, ecg_hidden: int = 512, ecg_layers: int = 12,
                 mlp_layers: int = 5, mlp_dropout: float = 0.2,
                 proj_dim: int = 256, fusion_layers: int = 5, fusion_heads: int = 8,
                 fusion_dropout: float = 0.1):
        super().__init__()
        if mode not in ("fusion", "text", "ecg"):
            raise ValueError(f"mode must be fusion/text/ecg, got: {mode}")
        self.mode = mode

        from transformers import BertModel

        self.text_encoder = BertModel.from_pretrained(macbert_path)
        self.text_proj = nn.Linear(768, proj_dim)          # text-CLS -> 256

        self.ecg_encoder = ECGWav2VecEncoder(num_leads, ecg_hidden, ecg_layers)
        self.ecg_mlp = ECGMlpProjection(ecg_hidden, mlp_layers, mlp_dropout, proj_dim)

        self.fusion_cls = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.fusion_layers = nn.ModuleList(
            [FusionTransformerLayer(proj_dim, fusion_heads, fusion_dropout)
             for _ in range(fusion_layers)]
        )
        self.classifier = nn.Linear(proj_dim, 1)           # single linear head

    def forward(self, ecg=None, input_ids=None, attention_mask=None):
        if self.mode in ("fusion", "text"):
            text_hidden = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            text_cls = text_hidden[:, 0]                   # text-CLS token (768)
            text_emb = self.text_proj(text_cls)            # (B, 256)
        else:
            text_emb = None

        if self.mode in ("fusion", "ecg"):
            ecg_emb = self.ecg_mlp(self.ecg_encoder(ecg))  # (B, 256)
        else:
            ecg_emb = None

        if self.mode == "text":
            return self.classifier(text_emb)               # (B, 1)

        if self.mode == "ecg":
            return self.classifier(ecg_emb)                # (B, 1)

        b = ecg_emb.size(0)
        tokens = torch.cat(
            [self.fusion_cls.expand(b, -1, -1), ecg_emb.unsqueeze(1), text_emb.unsqueeze(1)],
            dim=1,                                        # (B, 3, 256)
        )
        for layer in self.fusion_layers:
            tokens = layer(tokens)
        fused = tokens[:, 0]                               # fusion-CLS
        return self.classifier(fused)                      # (B, 1)
