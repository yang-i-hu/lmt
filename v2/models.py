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
5. TransXGBTransformer         – Transformer backbone for TransXGB two-stage model
6. CrossSectionalTransformer   – Attention across stocks within each date
7. TemporalTransformer         – Temporal attention over factor history per stock
8. SuperEncoder                – Autoencoder + cross-sectional attention on latent space
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
# 5. TransXGB Transformer  (backbone for Transformer + XGBoost stacking)
# ═════════════════════════════════════════════════════════════════════════

class TransXGBTransformer(nn.Module):
    """Transformer backbone for the TransXGB two-stage model.

    Architecture:
        factor values → value_proj + factor_embed → Transformer Encoder →
        CLS / mean pool → latent (d_model) → MLP head → prediction

    Same self-attention-over-factors design as FactorTransformer, but
    additionally exposes ``encode()`` and ``forward_with_latent()`` so the
    downstream XGBoost stage can access the pooled latent representation.

    Stage 1 (this module): trained via the standard tabular pipeline.
    Stage 2 (external):    XGBoost fitted on residuals using original
                           features + latent embeddings from this module.

    Input:  (B, n_factors)
    Output: (B,)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        pool: str = "cls",
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
            enable_nested_tensor=False,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enable_gradient_checkpointing(self):
        """Trade compute for memory — re-compute activations during backward."""
        self._use_checkpoint = True

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared path: project → embed → transformer → pool → latent (B, d_model)."""
        B = x.size(0)
        values = x.unsqueeze(-1)                               # (B, n_factors, 1)
        tokens = self.value_proj(values) + self.factor_embed   # (B, n_factors, d_model)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        if getattr(self, "_use_checkpoint", False):
            for layer in self.transformer.layers:
                tokens = torch.utils.checkpoint.checkpoint(layer, tokens, use_reentrant=False)
            h = tokens
        else:
            h = self.transformer(tokens)

        if self.pool == "cls":
            return h[:, 0, :]
        return h.mean(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled latent representation (B, d_model) without prediction head."""
        return self._transformer_forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard prediction: (B, n_factors) → (B,)."""
        rep = self._transformer_forward(x)
        return self.head(rep).squeeze(-1)

    def forward_with_latent(self, x: torch.Tensor):
        """Return both prediction and latent in a single pass.

        Returns
        -------
        pred : Tensor (B,)
        latent : Tensor (B, d_model)
        """
        rep = self._transformer_forward(x)
        pred = self.head(rep).squeeze(-1)
        return pred, rep


# ═════════════════════════════════════════════════════════════════════════
# 6. Cross-Sectional Transformer  (attention across stocks)
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
# 7. Temporal Transformer  (attention over time per stock)
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


# ═════════════════════════════════════════════════════════════════════════
# 8. SuperEncoder  (autoencoder + cross-sectional attention on latent)
# ═════════════════════════════════════════════════════════════════════════

class _ResidualBlockLN(nn.Module):
    """Residual block using LayerNorm (deterministic, no train/eval mismatch)."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class SuperEncoder(nn.Module):
    """Autoencoder with cross-sectional attention on the latent space.

    Combines the best of both worlds:
      - **Autoencoder bottleneck** compresses ~1000 collinear factors into a
        compact latent (e.g. 64-dim), providing strong regularisation.
      - **Cross-sectional transformer** applies multi-head self-attention
        across all stocks on the same date *in the latent space*, learning
        relative mispricing, sector structure, and crowding effects cheaply
        (O(N_stocks² × latent_dim)  instead of  O(N_stocks² × n_factors)).

    Architecture::

        Per-stock:   Input (F) → ResidualEncoder → latent_z  (64-dim)
        Per-date:    All latent_z → TransformerEncoder → enhanced_z
        Per-stock:   enhanced_z → Predictor → prediction
                     latent_z   → Decoder   → reconstruction (regulariser)

    Reconstruction is computed from the *pre-attention* latent so the
    encoder must learn a good factor representation on its own.

    Uses LayerNorm (not BatchNorm) for deterministic per-sample behaviour
    across dates — critical for cross-date IC stability (ICIR).

    Input:  (B_dates, N_stocks, n_factors)  with bool padding mask
    Output: (B_dates, N_stocks)
    """

    def __init__(
        self,
        input_size: int,
        encoder_sizes: List[int] = None,
        latent_dim: int = 64,
        predictor_sizes: List[int] = None,
        dropout: float = 0.3,
        recon_weight: float = 0.1,
        use_residual: bool = True,
        ic_loss_weight: float = 0.0,  # weight for IC consistency loss (0 = disabled)
        output_tanh: bool = True,     # cap predictions with tanh for stability
        cs_normalize: bool = True,    # cross-sectional z-score normalization on output
        # cross-sectional transformer params
        cs_n_heads: int = 4,
        cs_n_layers: int = 2,
        cs_dim_feedforward: int = 256,
        cs_dropout: float = None,     # separate dropout for CS attention (default: same as dropout)
    ):
        super().__init__()
        if encoder_sizes is None:
            encoder_sizes = [512, 256]
        if predictor_sizes is None:
            predictor_sizes = [128]

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.recon_weight = recon_weight
        self.ic_loss_weight = ic_loss_weight
        self.use_residual = use_residual
        self.output_tanh = output_tanh
        self.cs_normalize = cs_normalize

        # ── Encoder (per-stock, LayerNorm for cross-date stability) ──
        enc: list[nn.Module] = []
        prev = input_size
        for h in encoder_sizes:
            enc.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            if use_residual and prev == h:
                # wrap the last 4 layers as a residual block
                enc[-4:] = [_ResidualBlockLN(h, dropout)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        # ── Decoder (for reconstruction loss, mirrors encoder) ──
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(encoder_sizes):
            dec.extend([nn.Linear(prev, h), nn.GELU()])
            prev = h
        dec.append(nn.Linear(prev, input_size))
        self.decoder = nn.Sequential(*dec)

        # ── Cross-Sectional Transformer (on latent space) ──
        self.cs_norm = nn.LayerNorm(latent_dim)
        cs_drop = cs_dropout if cs_dropout is not None else dropout
        cs_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=cs_n_heads,
            dim_feedforward=cs_dim_feedforward,
            dropout=cs_drop,
            activation="gelu",
            batch_first=True,
        )
        self.cs_transformer = nn.TransformerEncoder(cs_layer, num_layers=cs_n_layers)

        # Learnable gate controlling cross-sectional contribution.
        # Initialised to 0.1 so the model starts mostly as a plain
        # autoencoder and must learn to trust the CS signal.
        self.cs_gate = nn.Parameter(torch.tensor(0.1))

        # ── Predictor head (per-stock, on enhanced latent) ──
        pred: list[nn.Module] = []
        prev = latent_dim
        for h in predictor_sizes:
            pred.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
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

    # ── helpers ──

    def _encode_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Encode a flat batch of samples: (N, F) → (N, latent_dim)."""
        return self.encoder(x_flat)

    def _decode_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode a flat batch: (N, latent_dim) → (N, F)."""
        return self.decoder(z_flat)

    # ── forward (cross-sectional batching) ──

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    (B, N_stocks, n_factors)
        mask: (B, N_stocks)  True = valid stock, False = padding
        Returns: (B, N_stocks) predictions
        """
        B, N, n_f = x.shape

        # 1. Per-stock encoding  (flatten → encode → reshape)
        x_flat = x.reshape(B * N, n_f)                            # (B*N, n_f)
        z_flat = self._encode_flat(x_flat)                         # (B*N, latent)
        z = z_flat.reshape(B, N, self.latent_dim)                  # (B, N, latent)

        # 2. Cross-sectional attention on latent representations
        z_normed = self.cs_norm(z)
        pad_mask = (~mask) if mask is not None else None
        h = self.cs_transformer(z_normed, src_key_padding_mask=pad_mask)  # (B, N, latent)

        # Gated residual: model learns how much to trust CS signal
        enhanced = torch.sigmoid(self.cs_gate) * h + z

        # 3. Per-stock prediction  (flatten → predict → reshape)
        enhanced_flat = enhanced.reshape(B * N, self.latent_dim)
        pred = self.predictor(enhanced_flat).squeeze(-1)           # (B*N,)
        pred = pred.reshape(B, N)                                  # (B, N)

        # 4. Output tanh capping — prevents outlier predictions
        if self.output_tanh:
            pred = torch.tanh(pred)

        # 5. Cross-sectional normalization — per-date z-score
        #    Forces predictions to have zero-mean, unit-variance per date,
        #    removing date-specific scale differences that destroy IC stability.
        if self.cs_normalize and mask is not None:
            for b in range(B):
                valid = mask[b]
                if valid.sum() < 2:
                    continue
                p = pred[b][valid]
                pred[b][valid] = (p - p.mean()) / (p.std() + 1e-8)

        return pred

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Combined prediction + reconstruction + IC consistency loss (masked).

        x:    (B, N_stocks, n_factors)
        y:    (B, N_stocks)
        mask: (B, N_stocks)  True = valid
        """
        B, N, n_f = x.shape

        # 1. Encode
        x_flat = x.reshape(B * N, n_f)
        z_flat = self._encode_flat(x_flat)
        z = z_flat.reshape(B, N, self.latent_dim)

        # 2. Cross-sectional attention + residual
        z_normed = self.cs_norm(z)
        pad_mask = (~mask) if mask is not None else None
        h = self.cs_transformer(z_normed, src_key_padding_mask=pad_mask)
        enhanced = torch.sigmoid(self.cs_gate) * h + z

        # 3. Prediction (includes optional tanh + CS normalization)
        enhanced_flat = enhanced.reshape(B * N, self.latent_dim)
        raw_pred = self.predictor(enhanced_flat).squeeze(-1).reshape(B, N)
        if self.output_tanh:
            raw_pred = torch.tanh(raw_pred)
        # For loss, use raw (non-CS-normalized) predictions against
        # CS-normalized targets to avoid information leakage in MSE
        pred_loss = criterion(raw_pred[mask], y[mask])
        # CS-normalize for IC loss (rank-aware)
        pred = raw_pred.clone()
        if self.cs_normalize:
            for b in range(B):
                valid = mask[b]
                if valid.sum() < 2:
                    continue
                p = pred[b][valid]
                pred[b][valid] = (p - p.mean()) / (p.std() + 1e-8)

        # 4. Reconstruction loss (from pre-attention latent, valid stocks only)
        total_loss = pred_loss
        if self.recon_weight > 0:
            x_recon = self._decode_flat(z_flat).reshape(B, N, n_f)
            recon_loss = F.mse_loss(x_recon[mask], x[mask])
            total_loss = total_loss + self.recon_weight * recon_loss

        # 5. IC consistency loss — penalise variance of per-date Pearson IC
        #    Uses differentiable Pearson correlation as a proxy for Spearman IC.
        #    Loss = -mean(IC_per_date) + std(IC_per_date)
        #    This directly optimises for higher, more stable IC → better ICIR.
        if self.ic_loss_weight > 0:
            date_ics = []
            for b in range(B):
                valid = mask[b]
                n_valid = valid.sum()
                if n_valid < 20:
                    continue
                p = pred[b][valid]
                t = y[b][valid]
                p_c = p - p.mean()
                t_c = t - t.mean()
                cov = (p_c * t_c).sum()
                denom = torch.sqrt((p_c ** 2).sum() * (t_c ** 2).sum() + 1e-8)
                date_ics.append(cov / denom)
            if len(date_ics) >= 2:
                ics = torch.stack(date_ics)
                # Maximise mean IC, minimise IC variance
                ic_loss = -ics.mean() + ics.std()
                total_loss = total_loss + self.ic_loss_weight * ic_loss

        return total_loss


# ═════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average) Model Wrapper
# ═════════════════════════════════════════════════════════════════════════

class EMAModel:
    """Maintains an exponential moving average of model parameters.

    The EMA model's predictions are smoother and more stable across dates
    than the raw training model — the NN equivalent of tree ensembling.
    Use the EMA model for inference / early-stopping evaluation.

    Usage::

        ema = EMAModel(model, decay=0.999)
        # In training loop:
        optimizer.step()
        ema.update()
        # For validation / OOS:
        ema.apply()      # swap EMA weights into model
        model.eval(); ...
        ema.restore()    # restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Call after each optimizer.step()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self):
        """Swap EMA weights into the model (for eval)."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original training weights (after eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return EMA shadow weights for saving."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        """Load EMA shadow weights."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()
