from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from src.inference.load_models import (
    get_device,
    load_vae,
    load_dcgan_generator,
    load_diffusion_model,
)
from src.inference.sample_grid import denormalize_images, save_tensor_grid
from src.utils.seed import set_seed


@torch.no_grad()
def generate_vae_samples(
    config: Dict,
    project_root: Path,
    num_images: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate samples from trained VAE.
    """
    set_seed(seed)
    device = get_device(config)

    model = load_vae(config, project_root, device)

    samples = model.sample(
        num_samples=num_images,
        device=device,
    )

    samples = denormalize_images(samples.detach().cpu())

    return samples


@torch.no_grad()
def generate_dcgan_samples(
    config: Dict,
    project_root: Path,
    num_images: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate samples from trained DCGAN generator.
    """
    set_seed(seed)
    device = get_device(config)

    generator = load_dcgan_generator(config, project_root, device)

    noise_dim = config["dcgan"]["noise_dim"]

    noise = torch.randn(
        num_images,
        noise_dim,
        1,
        1,
        device=device,
    )

    samples = generator(noise).detach().cpu()

    # DCGAN output is already in [-1, 1], convert to [0, 1]
    samples = denormalize_images(samples)

    return samples


@torch.no_grad()
def generate_diffusion_samples(
    config: Dict,
    project_root: Path,
    num_images: int = 64,
    seed: int = 42,
    inference_steps: int | None = None,
) -> torch.Tensor:
    """
    Generate samples from trained diffusion model.
    """
    set_seed(seed)
    device = get_device(config)

    model, scheduler = load_diffusion_model(config, project_root, device)

    diffusion_config = config["diffusion"]
    data_config = config["data"]

    image_size = diffusion_config["image_size"]
    channels = data_config["channels"]

    if inference_steps is None:
        inference_steps = diffusion_config["inference_steps"]

    images = torch.randn(
        num_images,
        channels,
        image_size,
        image_size,
        device=device,
    )

    scheduler.set_timesteps(inference_steps)

    for timestep in tqdm(
        scheduler.timesteps,
        desc="Generating diffusion samples",
    ):
        noise_pred = model(images, timestep).sample

        images = scheduler.step(
            noise_pred,
            timestep,
            images,
        ).prev_sample

    samples = denormalize_images(images.detach().cpu())

    return samples


def generate_and_save_all_models(
    config: Dict,
    project_root: Path,
    num_images: int = 64,
    seed: int = 42,
    inference_steps: int | None = None,
) -> None:
    """
    Generate and save sample grids from VAE, DCGAN, and diffusion models.
    """
    reports_config = config["reports"]

    vae_samples = generate_vae_samples(
        config=config,
        project_root=project_root,
        num_images=num_images,
        seed=seed,
    )

    save_tensor_grid(
        vae_samples,
        project_root / reports_config["vae_samples_grid"],
        nrow=8,
    )

    dcgan_samples = generate_dcgan_samples(
        config=config,
        project_root=project_root,
        num_images=num_images,
        seed=seed,
    )

    save_tensor_grid(
        dcgan_samples,
        project_root / reports_config["dcgan_samples_grid"],
        nrow=8,
    )

    diffusion_samples = generate_diffusion_samples(
        config=config,
        project_root=project_root,
        num_images=num_images,
        seed=seed,
        inference_steps=inference_steps,
    )

    save_tensor_grid(
        diffusion_samples,
        project_root / reports_config["diffusion_samples_grid"],
        nrow=8,
    )