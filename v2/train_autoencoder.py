"""
Factor Autoencoder — Rolling Per-Snapshot Training & Evaluation (v2)

Learns a compressed latent representation of the factor space before
predicting, providing better inductive bias for highly collinear factors.
Optional reconstruction loss regularises the latent space.

Usage:
    python train_autoencoder.py --config configs/autoencoder.yaml
    python train_autoencoder.py --config configs/autoencoder.yaml --device cuda:0
"""

from common import make_parser, run_tabular_pipeline
from models import FactorAutoencoder

DEFAULT_CONFIG = {
    "model": {
        "encoder_sizes": [512, 256],
        "latent_dim": 64,
        "predictor_sizes": [128],
        "dropout": 0.3,
        "recon_weight": 0.1,
    },
    "output": {"output_dir": "outputs_autoencoder"},
}


def model_factory(input_size: int, config: dict):
    mc = config.get("model", {})
    return FactorAutoencoder(
        input_size=input_size,
        encoder_sizes=mc.get("encoder_sizes", [512, 256]),
        latent_dim=mc.get("latent_dim", 64),
        predictor_sizes=mc.get("predictor_sizes", [128]),
        dropout=mc.get("dropout", 0.3),
        recon_weight=mc.get("recon_weight", 0.1),
    )


if __name__ == "__main__":
    args = make_parser(
        "Factor Autoencoder rolling per-snapshot training (v2)",
        "configs/autoencoder.yaml",
    ).parse_args()

    run_tabular_pipeline(
        config_path=args.config,
        device=args.device,
        snapshots_override=args.snapshots,
        universe_override=args.universe,
        default_config=DEFAULT_CONFIG,
        model_factory=model_factory,
        pipeline_name="Autoencoder",
        model_prefix="ae",
    )
