import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference.generate import (
    generate_vae_samples,
    generate_dcgan_samples,
    generate_diffusion_samples,
    generate_and_save_all_models,
)
from src.inference.sample_grid import save_tensor_grid
from src.utils.paths import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained generative models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["vae", "dcgan", "diffusion", "all"],
        help="Which model to generate samples from.",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=64,
        help="Number of images to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )

    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Number of diffusion inference steps.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    reports_config = config["reports"]

    if args.model == "all":
        generate_and_save_all_models(
            config=config,
            project_root=PROJECT_ROOT,
            num_images=args.num_images,
            seed=args.seed,
            inference_steps=args.inference_steps,
        )

    elif args.model == "vae":
        samples = generate_vae_samples(
            config=config,
            project_root=PROJECT_ROOT,
            num_images=args.num_images,
            seed=args.seed,
        )

        save_tensor_grid(
            samples,
            PROJECT_ROOT / reports_config["vae_samples_grid"],
            nrow=8,
        )

    elif args.model == "dcgan":
        samples = generate_dcgan_samples(
            config=config,
            project_root=PROJECT_ROOT,
            num_images=args.num_images,
            seed=args.seed,
        )

        save_tensor_grid(
            samples,
            PROJECT_ROOT / reports_config["dcgan_samples_grid"],
            nrow=8,
        )

    elif args.model == "diffusion":
        samples = generate_diffusion_samples(
            config=config,
            project_root=PROJECT_ROOT,
            num_images=args.num_images,
            seed=args.seed,
            inference_steps=args.inference_steps,
        )

        save_tensor_grid(
            samples,
            PROJECT_ROOT / reports_config["diffusion_samples_grid"],
            nrow=8,
        )


if __name__ == "__main__":
    main()