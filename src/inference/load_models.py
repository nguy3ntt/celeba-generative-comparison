from pathlib import Path
from typing import Dict

import torch
from diffusers import DDPMScheduler

from src.models.vae import ConvVAE
from src.models.dcgan import DCGANGenerator
from src.models.diffusion_unet import create_diffusion_unet


def get_device(config: Dict) -> torch.device:
    """
    Get available device from config.
    """
    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    return torch.device(device_name)


def load_vae(config: Dict, project_root: Path, device: torch.device) -> ConvVAE:
    """
    Load trained VAE checkpoint.
    """
    checkpoint_path = project_root / config["vae"]["checkpoint_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

    model = ConvVAE(
        image_channels=config["data"]["channels"],
        latent_dim=config["vae"]["latent_dim"],
        hidden_dims=config["vae"]["hidden_dims"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded VAE from: {checkpoint_path}")

    return model


def load_dcgan_generator(config: Dict, project_root: Path, device: torch.device) -> DCGANGenerator:
    """
    Load trained DCGAN generator checkpoint.
    """
    checkpoint_path = project_root / config["dcgan"]["checkpoint_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DCGAN checkpoint not found: {checkpoint_path}")

    model = DCGANGenerator(
        noise_dim=config["dcgan"]["noise_dim"],
        image_channels=config["data"]["channels"],
        features_g=config["dcgan"]["generator_features"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()

    print(f"Loaded DCGAN generator from: {checkpoint_path}")

    return model


def load_diffusion_model(config: Dict, project_root: Path, device: torch.device):
    """
    Load trained diffusion U-Net and DDPM scheduler.
    """
    checkpoint_path = project_root / config["diffusion"]["checkpoint_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")

    model = create_diffusion_unet(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=config["diffusion"]["timesteps"],
        beta_schedule="linear",
    )

    print(f"Loaded diffusion model from: {checkpoint_path}")

    return model, scheduler