"""
Model architectures for factor reweighting / cross-sectional alpha prediction.

All models output scalar predictions and are compatible with the training
pipelines in common.py or their respective custom training scripts.

Models
------
1. FactorMLP                   – Standard deep feedforward network (baseline)
2. ResidualMLP                 – ResNet-style residual blocks
3. FactorAutoencoder           – Encoder → latent → predictor (+ optional recon)
4. FactorTransformer           – Self-attention over factors (each factor = token)
5. CrossSectionalTransformer   – Attention across stocks within each date
6. TemporalTransformer         – Temporal attention over factor history per stock
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════
# 1. Standard MLP  (baseline — same architecture as train_dnn.py)
# ═════════════════════════════════════════════════════════════════════════

class FactorMLP(nn.Module):
    """Deep feedforward network for factor reweighting.

    Architecture:  [Linear → BN → Act → Dropout] × L  →  Linear(1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        dropout: float = 0.3,
        activation: str = "leaky_relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]

        act_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        act_fn = act_map.get(activation, nn.LeakyReLU(0.1))

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════
# 2. Residual MLP  (Deep Factor Network)
# ═════════════════════════════════════════════════════════════════════════

class _ResidualBlock(nn.Module):
    """Single residual block:  Linear → BN → GELU → Drop → Linear → BN  + skip."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResidualMLP(nn.Module):
    """ResNet-style deep factor network.

    Architecture:
        Input → Linear → BN → GELU → [ResidualBlock × N] → Linear → output

    Benefits over plain MLP:
      - Stabilises training for deep (10+ layer) networks
      - Better gradient flow for learning deep factor interactions
      - Easier to scale depth without degradation
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════
# 3. Factor Autoencoder  (nonlinear factor compression)
# ═════════════════════════════════════════════════════════════════════════

class FactorAutoencoder(nn.Module):
    """Autoencoder + Predictor for nonlinear factor compression.

    Architecture:
        Input → Encoder → Latent → Predictor → output
                       ↘ Decoder → reconstruction (optional loss)

    Many factors are highly collinear.  The encoder learns a compressed
    latent representation (e.g. 1000 → 64) with better inductive bias
    than predicting directly from raw factors.

    Set ``recon_weight > 0`` to add a reconstruction regulariser.
    """

    def __init__(
        self,
        input_size: int,
        encoder_sizes: List[int] = None,
        latent_dim: int = 64,
        predictor_sizes: List[int] = None,
        dropout: float = 0.3,
        recon_weight: float = 0.1,
    ):
        super().__init__()
        if encoder_sizes is None:
            encoder_sizes = [512, 256]
        if predictor_sizes is None:
            predictor_sizes = [128]

        self.recon_weight = recon_weight

        # ── Encoder ──
        enc = []
        prev = input_size
        for h in encoder_sizes:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        # ── Decoder (for reconstruction loss) ──
        dec = []
        prev = latent_dim
        for h in reversed(encoder_sizes):
            dec.extend([nn.Linear(prev, h), nn.GELU()])
            prev = h
        dec.append(nn.Linear(prev, input_size))
        self.decoder = nn.Sequential(*dec)

        # ── Predictor ──
        pred = []
        prev = latent_dim
        for h in predictor_sizes:
            pred.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        pred.append(nn.Linear(prev, 1))
        self.predictor = nn.Sequential(*pred)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.predictor(z).squeeze(-1)

    def compute_loss(self, x: torch.Tensor, y_true: torch.Tensor,
                     criterion: nn.Module) -> torch.Tensor:
        """Combined prediction + reconstruction loss."""
        z = self.encode(x)
        pred = self.predictor(z).squeeze(-1)
        pred_loss = criterion(pred, y_true)

        if self.recon_weight > 0:
            x_recon = self.decode(z)
            recon_loss = F.mse_loss(x_recon, x)
            return pred_loss + self.recon_weight * recon_loss
        return pred_loss


# ═════════════════════════════════════════════════════════════════════════
# 4. Factor Transformer  (self-attention over factors)
# ═════════════════════════════════════════════════════════════════════════

class FactorTransformer(nn.Module):
    """Self-attention over factors where each factor is one token.

    Architecture:
        factor values → value_proj + factor_embed → Transformer Encoder →
        CLS / mean pool → MLP head → prediction

    Explicitly models factor interactions (e.g. momentum ↔ volatility,
    value ↔ size) via attention rather than implicit MLP weights.

    ⚠ With N_factors tokens, self-attention is O(N²).  For >1000 factors
      consider reducing ``d_model`` or using mean pooling.
    """

    def __init__(
        self,
        input_size: int,          # number of factors
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        pool: str = "cls",        # "cls" or "mean"
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.pool = pool

        # Project each scalar factor value to d_model
        self.value_proj = nn.Linear(1, d_model)

        # Learned factor position embeddings
        self.factor_embed = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)

        # CLS token
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,  # avoid shape issues with gradient checkpointing
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def enable_gradient_checkpointing(self):
        """Trade compute for memory — re-compute activations during backward."""
        self._use_checkpoint = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_factors)
        B = x.size(0)

        values = x.unsqueeze(-1)                               # (B, n_factors, 1)
        tokens = self.value_proj(values) + self.factor_embed   # (B, n_factors, d_model)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)           # (B, 1+n_factors, d_model)

        # Apply transformer with optional gradient checkpointing
        if getattr(self, "_use_checkpoint", False):
            for layer in self.transformer.layers:
                tokens = torch.utils.checkpoint.checkpoint(layer, tokens, use_reentrant=False)
            h = tokens
        else:
            h = self.transformer(tokens)

        if self.pool == "cls":
            rep = h[:, 0, :]          # CLS token
        else:
            rep = h.mean(dim=1)       # mean pool

        return self.head(rep).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════
# 5. Cross-Sectional Transformer  (attention across stocks)
# ═════════════════════════════════════════════════════════════════════════

class CrossSectionalTransformer(nn.Module):
    """Attention across stocks within each date.

    Architecture:
        [stock1 factors, stock2 factors, …]
            → stock embedding (Linear)
            → Transformer across stocks
            → per-stock MLP head
            → prediction per stock

    Learns:
      - Relative mispricing
      - Sector structure
      - Crowding / cross-sectional dependencies

    Input:  (B_dates, N_stocks, n_factors)  with a bool padding mask
    Output: (B_dates, N_stocks)
    """

    def __init__(
        self,
        n_factors: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_factors = n_factors

        self.stock_proj = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    (B, N_stocks, n_factors)
        mask: (B, N_stocks)  True = valid stock, False = padding
        """
        h = self.stock_proj(x)                                     # (B, N, d_model)

        # Transformer expects True = *ignore* in src_key_padding_mask
        pad_mask = (~mask) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=pad_mask)     # (B, N, d_model)

        return self.head(h).squeeze(-1)                            # (B, N)


# ═════════════════════════════════════════════════════════════════════════
# 6. Temporal Transformer  (attention over time per stock)
# ═════════════════════════════════════════════════════════════════════════

class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (handles even & odd d_model)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        cos_vals = torch.cos(pos * div)
        pe[:, 1::2] = cos_vals[:, : pe[:, 1::2].size(1)]
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """Temporal attention over a sliding window of factor history per stock.

    Architecture:
        [factors_{t-W}, …, factors_t]
            → input projection + positional encoding
            → causal Transformer
            → last time-step representation
            → MLP head → prediction

    Models temporal evolution: momentum decay, volatility clustering,
    regime shifts.

    Input:  (B, window_size, n_factors)
    Output: (B,)
    """

    def __init__(
        self,
        n_factors: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        window_size: int = 20,
        pool: str = "last",       # "last" or "mean"
    ):
        super().__init__()
        self.window_size = window_size
        self.pool = pool

        self.input_proj = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoding = _SinusoidalPE(d_model, max_len=window_size + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pre-build causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(window_size, window_size), diagonal=1).bool(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window_size, n_factors)"""
        W = x.size(1)
        h = self.input_proj(x)           # (B, W, d_model)
        h = self.pos_encoding(h)         # (B, W, d_model)

        mask = self.causal_mask[:W, :W]
        h = self.transformer(h, mask=mask)

        if self.pool == "last":
            rep = h[:, -1, :]
        else:
            rep = h.mean(dim=1)

        return self.head(rep).squeeze(-1)
