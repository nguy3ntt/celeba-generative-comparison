import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import build_model_summary
from src.evaluation.nearest_neighbors import run_diffusion_nearest_neighbor_check
from src.evaluation.visualize import create_model_comparison_figure
from src.utils.paths import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained generative models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--nearest-neighbor",
        action="store_true",
        help="Run nearest-neighbour check for diffusion samples.",
    )

    parser.add_argument(
        "--num-generated",
        type=int,
        default=8,
        help="Number of generated images for nearest-neighbour check.",
    )

    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=3,
        help="Number of nearest real images per generated image.",
    )

    parser.add_argument(
        "--max-real-images",
        type=int,
        default=5000,
        help="Number of real training images used for nearest-neighbour bank.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(PROJECT_ROOT / args.config)

    print("Building model summary...")
    summary_df = build_model_summary(
        config=config,
        project_root=PROJECT_ROOT,
    )

    print(summary_df)

    print("Creating model comparison figure...")
    create_model_comparison_figure(
        image_paths={
            "VAE": PROJECT_ROOT / config["reports"]["vae_samples_grid"],
            "DCGAN": PROJECT_ROOT / config["reports"]["dcgan_samples_grid"],
            "Diffusion": PROJECT_ROOT / config["reports"]["diffusion_samples_grid"],
        },
        output_path=PROJECT_ROOT / config["reports"]["model_comparison_grid"],
    )

    if args.nearest_neighbor:
        print("Running nearest-neighbour check...")
        run_diffusion_nearest_neighbor_check(
            config=config,
            project_root=PROJECT_ROOT,
            num_generated=args.num_generated,
            num_neighbors=args.num_neighbors,
            max_real_images=args.max_real_images,
        )


if __name__ == "__main__":
    main()