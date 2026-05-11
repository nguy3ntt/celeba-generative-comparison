from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import CelebAProcessedDataset
from src.models.diffusion_unet import create_diffusion_unet
from src.utils.paths import ensure_dir


def get_diffusion_transforms(config: Dict):
    """
    Transform processed PIL images into normalized tensors
    """
    data_config = config["data"]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config["normalize_mean"],
                std=data_config["normalize_std"],
            ),
        ]
    )


def create_diffusion_dataloader(
    config: Dict,
    project_root: Path,
    limit: Optional[int] = None,
):
    """
    Create train dataloader for diffusion training
    """
    transform = get_diffusion_transforms(config)

    manifest_path = project_root / "data" / "interim" / "celeba_manifest.csv"

    train_dataset = CelebAProcessedDataset(
        manifest_path=manifest_path,
        split="train",
        transform=transform,
        project_root=project_root,
        limit=limit,
    )

    diffusion_config = config["diffusion"]
    data_config = config["data"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=diffusion_config.get("batch_size", data_config["batch_size"]),
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        drop_last=True,
    )

    return train_loader


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1, 1] to [0, 1] for saving.
    """
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def sample_diffusion_images(
    model,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_images: int = 64,
    image_size: int = 64,
    channels: int = 3,
    inference_steps: int = 50,
):
    """
    Generate images by starting from random noise and denoising step by step
    """
    model.eval()

    images = torch.randn(
        num_images,
        channels,
        image_size,
        image_size,
        device=device,
    )

    noise_scheduler.set_timesteps(inference_steps)

    for timestep in tqdm(
        noise_scheduler.timesteps,
        desc="Sampling diffusion images",
        leave=False,
    ):
        model_output = model(images, timestep).sample
        images = noise_scheduler.step(
            model_output,
            timestep,
            images,
        ).prev_sample

    images = denormalize_images(images.detach().cpu())

    model.train()

    return images


@torch.no_grad()
def save_denoising_progression(
    model,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    output_path: Path,
    image_size: int = 64,
    channels: int = 3,
    inference_steps: int = 50,
):
    """
    Save one image's denoising progression from random noise to final sample
    """
    model.eval()

    image = torch.randn(
        1,
        channels,
        image_size,
        image_size,
        device=device,
    )

    noise_scheduler.set_timesteps(inference_steps)

    saved_steps = []
    save_indices = {
        0,
        int(inference_steps * 0.2),
        int(inference_steps * 0.4),
        int(inference_steps * 0.6),
        int(inference_steps * 0.8),
        inference_steps - 1,
    }

    for i, timestep in enumerate(noise_scheduler.timesteps):
        model_output = model(image, timestep).sample
        image = noise_scheduler.step(
            model_output,
            timestep,
            image,
        ).prev_sample

        if i in save_indices:
            saved_steps.append(denormalize_images(image.detach().cpu())[0])

    if len(saved_steps) > 0:
        ensure_dir(output_path.parent)
        grid = torch.stack(saved_steps)

        save_image(
            grid,
            output_path,
            nrow=len(saved_steps),
        )

    model.train()


def save_diffusion_samples(
    model,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    output_path: Path,
    config: Dict,
):
    """
    Generate and save a grid of diffusion samples
    """
    diffusion_config = config["diffusion"]
    data_config = config["data"]

    samples = sample_diffusion_images(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        num_images=64,
        image_size=diffusion_config["image_size"],
        channels=data_config["channels"],
        inference_steps=diffusion_config["inference_steps"],
    )

    ensure_dir(output_path.parent)

    save_image(
        samples,
        output_path,
        nrow=8,
    )


def train_diffusion(
    config: Dict,
    project_root: Path,
    epochs: Optional[int] = None,
    limit: Optional[int] = None,
):
    """
    Full DDPM-style diffusion training loop.

    The model is trained to predict the noise added to real images.
    """
    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)
    print(f"Using device: {device}")

    train_loader = create_diffusion_dataloader(
        config=config,
        project_root=project_root,
        limit=limit,
    )

    diffusion_config = config["diffusion"]

    model = create_diffusion_unet(config).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_config["timesteps"],
        beta_schedule="linear",
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=diffusion_config["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_epochs = epochs if epochs is not None else diffusion_config["epochs"]

    checkpoint_dir = ensure_dir(project_root / "checkpoints" / "diffusion")
    runs_dir = ensure_dir(project_root / "runs" / "diffusion")
    generated_dir = ensure_dir(project_root / diffusion_config["outputs"]["generated_dir"])
    denoising_dir = ensure_dir(project_root / diffusion_config["outputs"]["denoising_progress_dir"])

    training_log = []

    print("Diffusion training settings:")
    print(f"  image size: {diffusion_config['image_size']}")
    print(f"  train timesteps: {diffusion_config['timesteps']}")
    print(f"  inference steps: {diffusion_config['inference_steps']}")
    print(f"  learning rate: {diffusion_config['learning_rate']}")
    print(f"  batch size: {diffusion_config.get('batch_size', config['data']['batch_size'])}")

    for epoch in range(1, num_epochs + 1):
        model.train()

        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=True,
        )

        for clean_images in progress_bar:
            clean_images = clean_images.to(device)

            batch_size = clean_images.size(0)

            noise = torch.randn_like(clean_images)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(
                clean_images,
                noise,
                timesteps,
            )

            noise_pred = model(
                noisy_images,
                timesteps,
            ).sample

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

        epoch_loss /= len(train_loader)

        log_row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "learning_rate": diffusion_config["learning_rate"],
            "timesteps": diffusion_config["timesteps"],
            "inference_steps": diffusion_config["inference_steps"],
        }

        training_log.append(log_row)

        print(f"Epoch {epoch}: train_loss={epoch_loss:.6f}")

        if epoch == 1 or epoch % config["training"]["save_every"] == 0 or epoch == num_epochs:
            sample_path = generated_dir / f"epoch_{epoch:03d}_generated.png"

            save_diffusion_samples(
                model=model,
                noise_scheduler=noise_scheduler,
                device=device,
                output_path=sample_path,
                config=config,
            )

            progression_path = denoising_dir / f"epoch_{epoch:03d}_denoising_progress.png"

            save_denoising_progression(
                model=model,
                noise_scheduler=noise_scheduler,
                device=device,
                output_path=progression_path,
                image_size=diffusion_config["image_size"],
                channels=config["data"]["channels"],
                inference_steps=diffusion_config["inference_steps"],
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "scheduler_config": noise_scheduler.config,
                },
                checkpoint_dir / f"diffusion_epoch_{epoch:03d}.pth",
            )

    final_checkpoint_path = project_root / diffusion_config["checkpoint_path"]
    ensure_dir(final_checkpoint_path.parent)

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "scheduler_config": noise_scheduler.config,
        },
        final_checkpoint_path,
    )

    log_df = pd.DataFrame(training_log)
    log_path = runs_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"Saved final diffusion checkpoint to: {final_checkpoint_path}")
    print(f"Saved diffusion training log to: {log_path}")

    report_sample_path = project_root / config["reports"]["diffusion_samples_grid"]

    save_diffusion_samples(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        output_path=report_sample_path,
        config=config,
    )

    print(f"Saved diffusion sample grid to: {report_sample_path}")

    return model, log_df