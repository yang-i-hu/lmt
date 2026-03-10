"""
Residual MLP — Rolling Per-Snapshot Training & Evaluation (v2)

ResNet-style deep factor network with skip connections.  Stabilises
training for deeper architectures and improves generalisation on
noisy high-dimensional factor data.

Usage:
    python train_residual_mlp.py --config configs/residual_mlp.yaml
    python train_residual_mlp.py --config configs/residual_mlp.yaml --device cuda:0
    python train_residual_mlp.py --config configs/residual_mlp.yaml --universe ../universe.txt
"""

from common import make_parser, run_tabular_pipeline
from models import ResidualMLP

# ── Model-specific defaults (merged with BASE_DEFAULT_CONFIG) ──────────

DEFAULT_CONFIG = {
    "model": {
        "hidden_dim": 256,
        "n_blocks": 4,
        "dropout": 0.3,
    },
    "output": {"output_dir": "outputs_residual_mlp"},
}


def model_factory(input_size: int, config: dict):
    mc = config.get("model", {})
    return ResidualMLP(
        input_size=input_size,
        hidden_dim=mc.get("hidden_dim", 256),
        n_blocks=mc.get("n_blocks", 4),
        dropout=mc.get("dropout", 0.3),
    )


if __name__ == "__main__":
    args = make_parser(
        "Residual MLP rolling per-snapshot training (v2)",
        "configs/residual_mlp.yaml",
    ).parse_args()

    run_tabular_pipeline(
        config_path=args.config,
        device=args.device,
        snapshots_override=args.snapshots,
        universe_override=args.universe,
        default_config=DEFAULT_CONFIG,
        model_factory=model_factory,
        pipeline_name="Residual MLP",
        model_prefix="resmlp",
    )
