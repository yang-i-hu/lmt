"""
Factor Transformer — Rolling Per-Snapshot Training & Evaluation (v2)

Treats each factor as a token and applies self-attention, explicitly
learning factor interactions (momentum ↔ volatility, value ↔ size, etc.)
rather than relying on implicit MLP weights.

Usage:
    python train_factor_transformer.py --config configs/factor_transformer.yaml
    python train_factor_transformer.py --config configs/factor_transformer.yaml --device cuda:0
"""

from common import make_parser, run_tabular_pipeline
from models import FactorTransformer

DEFAULT_CONFIG = {
    "model": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.3,
        "pool": "cls",
    },
    "training": {
        "batch_size": 128,        # small batch: each sample = N_factor tokens → O(N²) attention
        "learning_rate": 0.00005,  # very low — transformer + noisy labels
        "weight_decay": 0.01,
        "warmup_epochs": 10,
    },
    "output": {"output_dir": "outputs_factor_transformer"},
}


def model_factory(input_size: int, config: dict):
    mc = config.get("model", {})
    return FactorTransformer(
        input_size=input_size,
        d_model=mc.get("d_model", 64),
        n_heads=mc.get("n_heads", 4),
        n_layers=mc.get("n_layers", 2),
        dim_feedforward=mc.get("dim_feedforward", 256),
        dropout=mc.get("dropout", 0.3),
        pool=mc.get("pool", "cls"),
    )


if __name__ == "__main__":
    args = make_parser(
        "Factor Transformer rolling per-snapshot training (v2)",
        "configs/factor_transformer.yaml",
    ).parse_args()

    run_tabular_pipeline(
        config_path=args.config,
        device=args.device,
        snapshots_override=args.snapshots,
        universe_override=args.universe,
        default_config=DEFAULT_CONFIG,
        model_factory=model_factory,
        pipeline_name="Factor Transformer",
        model_prefix="ftrans",
    )
