# Model Architectures

This document describes the 7 models available in the factor reweighting pipeline.
All models solve the same task: **predict a stock's forward 10-day return from its
factor exposures**, trained independently per snapshot in a rolling fashion.

---

## Overview

| # | Model | Type | Input Shape | Script | What It Learns |
|---|-------|------|-------------|--------|----------------|
| 0 | ElasticNet | Linear | `(N, F)` | `train_elasticnet.py` | Sparse linear factor weights |
| 1 | DNN (MLP) | Feedforward NN | `(N, F)` | `train_dnn.py` | Nonlinear factor combinations |
| 2 | Residual MLP | Deep ResNet-style | `(N, F)` | `train_residual_mlp.py` | Deep nonlinear interactions with skip connections |
| 3 | Factor Autoencoder | Encoder–Predictor | `(N, F)` | `train_autoencoder.py` | Compressed latent factor representation |
| 4 | Factor Transformer | Self-attention | `(N, F)` | `train_factor_transformer.py` | Explicit factor↔factor interactions |
| 5 | Cross-Sectional Transformer | Cross-stock attention | `(D, S, F)` | `train_cross_sectional.py` | Relative mispricing across stocks per date |
| 6 | Temporal Transformer | Causal temporal attention | `(N, W, F)` | `train_temporal.py` | Factor momentum, regime shifts over time |

> `N` = samples (stock×date rows), `F` = number of factors, `D` = dates,
> `S` = stocks per date, `W` = window size (historical dates).

---

## Loss Functions

### Primary loss — all models

$$\mathcal{L}_{\text{pred}} = \text{MSE}(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

where $y_i$ is the forward 10-day label (`labelValue` from Label10.h5)
and $\hat{y}_i$ is the model prediction.

### ElasticNet — built-in regularised loss

$$\mathcal{L}_{\text{elasticnet}} = \frac{1}{2N}\|y - X\beta\|_2^2
  + \alpha \left( \frac{1-\rho}{2}\|\beta\|_2^2 + \rho\|\beta\|_1 \right)$$

- $\alpha$ = overall regularisation strength (default `0.01`)
- $\rho$ = `l1_ratio` (default `0.5`): balances L1 (sparsity) vs L2 (shrinkage)
- $\rho = 0$ → Ridge, $\rho = 1$ → Lasso, $\rho = 0.5$ → balanced

### Factor Autoencoder — composite loss

$$\mathcal{L}_{\text{autoencoder}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{recon}} \cdot \text{MSE}(x, \hat{x})$$

where $\hat{x} = \text{Decoder}(\text{Encoder}(x))$ is the reconstructed input.
The reconstruction term ($\lambda_{\text{recon}}$ = `recon_weight`, default `0.1`)
regularises the latent space so the encoder preserves factor structure, not just
whatever minimises prediction loss.

### All deep learning models — additional regularisation

| Technique | Setting | Purpose |
|-----------|---------|---------|
| **AdamW weight decay** | `weight_decay` in config | L2 regularisation decoupled from gradient |
| **Dropout** | `dropout` (default 0.3) | Prevents co-adaptation of neurons |
| **Gradient clipping** | `max_norm = 1.0` | Prevents exploding gradients |
| **Early stopping** | `patience = 15` epochs | Stops when val loss stops improving, restores best weights |
| **BatchNorm / LayerNorm** | Per-layer | Stabilises training, acts as mild regulariser |

---

## Model Details

### 0. ElasticNet

**Type:** Linear model (scikit-learn)

```
Input (F factors) → Linear combination → scalar prediction
```

- No GPU required — trained on CPU via coordinate descent
- Interpretable: each coefficient is a factor weight
- Produces sparsity analysis (% zero coefficients, top features)
- Best for: baseline, understanding which factors matter

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.01 | Higher → stronger regularisation → sparser model |
| `l1_ratio` | 0.5 | 1.0 = pure Lasso (sparse), 0.0 = pure Ridge (dense) |
| `max_iter` | 2000 | Convergence iterations |

---

### 1. DNN (FactorMLP)

**Type:** Deep feedforward network

```
Input (F) → [Linear → BatchNorm → LeakyReLU → Dropout] × 4 → Linear(1)
```

Default architecture: `512 → 256 → 128 → 64 → 1`

- Kaiming initialisation for all linear layers
- Configurable activation function (`relu`, `leaky_relu`, `elu`, `gelu`, `tanh`)
- Learns arbitrary nonlinear factor combinations
- Best for: capturing nonlinear factor interactions when you have enough data

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hidden_sizes` | [512, 256, 128, 64] | Width and depth of network |
| `dropout` | 0.3 | Regularisation strength |
| `activation` | leaky_relu | Nonlinearity type |
| `batch_norm` | true | Stabilises training |

---

### 2. Residual MLP

**Type:** ResNet-style deep feedforward network

```
Input (F) → Linear → BN → GELU → [ResidualBlock × N] → Linear(1)

ResidualBlock:
    x → Linear → BN → GELU → Dropout → Linear → BN → (+x) → GELU
```

- Skip connections allow training much deeper networks (10+ layers)
- Better gradient flow than plain MLP
- Each block has 2 linear layers + skip, so effective depth ≈ 2×`n_blocks` + 2
- Best for: when plain MLP is underfitting and you want more depth

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hidden_dim` | 256 | Width of all residual blocks (fixed) |
| `n_blocks` | 4 | Number of residual blocks (depth) |
| `dropout` | 0.3 | Dropout inside each block |

---

### 3. Factor Autoencoder

**Type:** Encoder → Latent → Predictor (with optional reconstruction)

```
Input (F) → Encoder [512 → 256] → Latent (64) → Predictor [128] → scalar
                                 ↘ Decoder [256 → 512] → Reconstruction (F)
```

- Compresses ~1000 correlated factors into a compact latent space (e.g. 64 dims)
- The predictor operates on the compressed representation
- Reconstruction loss acts as a regulariser — forces the latent space to retain
  useful factor structure rather than collapsing to only prediction-relevant info
- Best for: highly collinear factor spaces where dimensionality reduction helps

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `encoder_sizes` | [512, 256] | Encoder hidden layers |
| `latent_dim` | 64 | Bottleneck dimension |
| `predictor_sizes` | [128] | Prediction head hidden layers |
| `recon_weight` | 0.1 | Weight of reconstruction loss (0 = disabled) |
| `dropout` | 0.3 | Dropout in encoder and predictor |

---

### 4. Factor Transformer

**Type:** Self-attention where each factor is a token

```
Input (F scalars) → unsqueeze → value_proj(1→d) + factor_embed
    → [CLS] + factor tokens (F+1 tokens)
    → TransformerEncoder (L layers, H heads)
    → CLS token representation
    → MLP head → scalar
```

- Each factor value is projected to `d_model` dimensions and added to a
  learned factor positional embedding
- Self-attention explicitly models factor↔factor interactions
  (e.g. momentum ↔ volatility, value ↔ size)
- ⚠️ **Memory:** $O(F^2)$ attention per sample — with 1000+ factors,
  batch size must be small
- Supports gradient checkpointing to trade compute for memory
- Best for: discovering which factor interactions matter most

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `d_model` | 64 | Token embedding dimension |
| `n_heads` | 4 | Attention heads (must divide `d_model`) |
| `n_layers` | 2 | Transformer encoder layers |
| `dim_feedforward` | 128 | FFN hidden dim inside each layer |
| `pool` | "cls" | Aggregation: "cls" token or "mean" pool |
| `gradient_checkpointing` | true | Recompute activations to save GPU memory |

---

### 5. Cross-Sectional Transformer

**Type:** Attention across stocks within each date

```
Per date:
    [stock₁ factors, stock₂ factors, …, stockₛ factors]   (S, F)
        → stock_proj (F → d_model)
        → TransformerEncoder across stocks
        → per-stock MLP head
        → prediction per stock                              (S,)
```

- Each "sample" is one **date** containing all stocks
- Attention operates across the stock dimension — each stock sees all others
- Learns relative mispricing, sector structure, crowding effects
- Padding mask handles variable number of stocks per date
- ⚠️ This is **not** a tabular model — it has a custom training pipeline
- Best for: cross-sectional alpha where relative stock positioning matters

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `d_model` | 128 | Stock embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Transformer layers |
| `dim_feedforward` | 512 | FFN dimension |
| `dropout` | 0.3 | Dropout |

---

### 6. Temporal Transformer

**Type:** Causal temporal attention over factor history per stock

```
Per stock:
    [factors_{t-W}, factors_{t-W+1}, …, factors_t]   (W, F)
        → input_proj (F → d_model) + sinusoidal PE
        → Causal TransformerEncoder (masked: no future leakage)
        → last time-step representation
        → MLP head → scalar prediction for date t
```

- Sliding window of `W` historical dates per stock (default W=20)
- Causal mask ensures each time step only attends to past/present
- Sinusoidal positional encoding for temporal ordering
- Uses tail of IS data as history context for early OOS dates
- ⚠️ This is **not** a tabular model — it has a custom training pipeline
- Best for: capturing momentum decay, volatility clustering, regime changes

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `d_model` | 128 | Time-step embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Transformer layers |
| `dim_feedforward` | 512 | FFN dimension |
| `window_size` | 20 | Historical dates as context |
| `pool` | "last" | "last" (causal) or "mean" aggregation |

---

## Training Configuration

All deep learning models share these training settings:

| Parameter | MLP / ResNet / AE | Transformers | Purpose |
|-----------|-------------------|-------------|---------|
| `learning_rate` | 0.0001 | 0.00005 | Peak LR after warmup |
| `weight_decay` | 0.001 | 0.01 | AdamW L2 regularisation |
| `batch_size` | 512 | 128–512 | Samples per gradient step |
| `warmup_epochs` | 5 | 10 | Linear LR warmup before cosine decay |
| `epochs` | 100 | 100 | Maximum training epochs |
| `early_stopping_patience` | 15 | 15 | Epochs without improvement before stop |
| `val_ratio` | 0.15 | 0.15 | Last 15% of IS dates for validation |

### LR Schedule

```
LR
 ↑
 │           peak LR
 │         ╱‾‾‾‾‾‾╲
 │       ╱          ╲
 │     ╱              ╲
 │   ╱    cosine        ╲
 │ ╱      decay           ╲
 │╱                         ╲___
 └──────────────────────────────→ epoch
   warmup    cosine annealing
```

1. **Warmup phase** (LinearLR): LR ramps from 1% → 100% of target over `warmup_epochs`
2. **Cosine decay** (CosineAnnealingLR): LR decays from peak to ~0 over remaining epochs

### Why these settings for financial data

Financial return prediction has an extremely low signal-to-noise ratio.
Aggressive learning rates cause the model to overfit noise in the first
few epochs. The combination of:

- **Low LR + warmup** → prevents overshooting the faint signal
- **Strong weight decay** → penalises large weights that fit noise
- **Small batch size** → more gradient updates per epoch, smoother convergence
- **Early stopping** → halts when generalisation stops improving

---

## Evaluation

All models are evaluated using the **LMT Data API**:

| Metric | Function | Description |
|--------|----------|-------------|
| **IC** (Information Coefficient) | `da_eva_ic()` | Spearman rank correlation between predictions and forward returns |
| **Group Return** | `da_eva_group_return()` | Return of long-short quintile portfolios |
| **Group IR** | (from above) | Information ratio of group returns |
| **Group Hit Score** | (from above) | Consistency of group ordering |

Evaluation runs both **per-snapshot** and **aggregated** across all snapshots.
Results are saved as CSVs in each run's output directory.
