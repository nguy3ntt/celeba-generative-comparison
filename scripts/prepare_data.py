import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.prepare_celeba import prepare_celeba
from src.utils.paths import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CelebA dataset.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing. Example: --limit 1000",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    prepare_celeba(config=config, limit=args.limit)


if __name__ == "__main__":
    main()