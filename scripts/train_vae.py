import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.training.train_vae import train_vae
from src.utils.paths import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE baseline on CelebA.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of VAE training epochs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional dataset limit for quick testing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    set_seed(config["project"]["seed"])

    train_vae(
        config=config,
        project_root=PROJECT_ROOT,
        epochs=args.epochs,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()