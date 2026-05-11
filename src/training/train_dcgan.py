from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import CelebAProcessedDataset
from src.models.dcgan import (
    DCGANGenerator,
    DCGANDiscriminator,
    weights_init_normal,
)
from src.utils.paths import ensure_dir


def get_dcgan_transforms(config: Dict):
    """
    Transform processed PIL images into normalized tensors

    DCGAN uses image values in [-1, 1] because the generator output uses Tanh
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


def create_dcgan_dataloader(
    config: Dict,
    project_root: Path,
    limit: Optional[int] = None,
):
    """
    Create train dataloader for DCGAN
    """
    transform = get_dcgan_transforms(config)

    manifest_path = project_root / "data" / "interim" / "celeba_manifest.csv"

    train_dataset = CelebAProcessedDataset(
        manifest_path=manifest_path,
        split="train",
        transform=transform,
        project_root=project_root,
        limit=limit,
    )

    data_config = config["data"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        drop_last=True,
    )

    return train_loader


def get_instance_noise_std(
    epoch: int,
    num_epochs: int,
    start_std: float,
    end_std: float,
) -> float:
    """
    Linearly decay instance noise from start_std to end_std
    """
    if num_epochs <= 1:
        return end_std

    progress = (epoch - 1) / (num_epochs - 1)
    current_std = start_std + progress * (end_std - start_std)

    return max(current_std, end_std)


def add_instance_noise(images: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Add small Gaussian noise to images before passing them to discriminator

    This helps prevent the discriminator from becoming too confident too early
    """
    if noise_std <= 0:
        return images

    noise = torch.randn_like(images) * noise_std
    noisy_images = images + noise

    return torch.clamp(noisy_images, -1.0, 1.0)


def save_dcgan_samples(
    generator: DCGANGenerator,
    fixed_noise: torch.Tensor,
    output_path: Path,
):
    """
    Save generated image grid using fixed noise

    Images are generated in [-1, 1], so save_image uses normalize=True
    """
    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()

    ensure_dir(output_path.parent)

    save_image(
        fake_images,
        output_path,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    generator.train()


def train_dcgan(
    config: Dict,
    project_root: Path,
    epochs: Optional[int] = None,
    limit: Optional[int] = None,
):
    """
    Full DCGAN training loop

    Stabilisation used:
    - one-sided label smoothing
    - lower discriminator learning rate
    - instance noise on discriminator inputs
    """
    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)
    print(f"Using device: {device}")

    train_loader = create_dcgan_dataloader(
        config=config,
        project_root=project_root,
        limit=limit,
    )

    data_config = config["data"]
    dcgan_config = config["dcgan"]

    noise_dim = dcgan_config["noise_dim"]

    generator = DCGANGenerator(
        noise_dim=noise_dim,
        image_channels=data_config["channels"],
        features_g=dcgan_config["generator_features"],
    ).to(device)

    discriminator = DCGANDiscriminator(
        image_channels=data_config["channels"],
        features_d=dcgan_config["discriminator_features"],
    ).to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    criterion = nn.BCEWithLogitsLoss()

    g_lr = dcgan_config.get("generator_learning_rate", dcgan_config.get("learning_rate", 0.0002))
    d_lr = dcgan_config.get("discriminator_learning_rate", dcgan_config.get("learning_rate", 0.0002))

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=g_lr,
        betas=(dcgan_config["beta1"], dcgan_config["beta2"]),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=d_lr,
        betas=(dcgan_config["beta1"], dcgan_config["beta2"]),
    )

    num_epochs = epochs if epochs is not None else dcgan_config["epochs"]

    checkpoint_dir = ensure_dir(project_root / "checkpoints" / "dcgan")
    runs_dir = ensure_dir(project_root / "runs" / "dcgan")
    fixed_noise_dir = ensure_dir(project_root / dcgan_config["outputs"]["fixed_noise_dir"])

    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)

    real_label_smooth = dcgan_config.get("real_label_smooth", 0.9)
    fake_label = dcgan_config.get("fake_label", 0.0)

    instance_noise_start = dcgan_config.get("instance_noise_start", 0.05)
    instance_noise_end = dcgan_config.get("instance_noise_end", 0.0)

    print("DCGAN tuning settings:")
    print(f"  generator lr: {g_lr}")
    print(f"  discriminator lr: {d_lr}")
    print(f"  real label smoothing: {real_label_smooth}")
    print(f"  fake label: {fake_label}")
    print(f"  instance noise: {instance_noise_start} -> {instance_noise_end}")

    training_log = []

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0

        noise_std = get_instance_noise_std(
            epoch=epoch,
            num_epochs=num_epochs,
            start_std=instance_noise_start,
            end_std=instance_noise_end,
        )

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=True,
        )

        for real_images in progress_bar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # train discriminator

            discriminator.zero_grad(set_to_none=True)

            real_targets = torch.full(
                (batch_size,),
                real_label_smooth,
                dtype=torch.float,
                device=device,
            )

            fake_targets = torch.full(
                (batch_size,),
                fake_label,
                dtype=torch.float,
                device=device,
            )

            # Real images with small instance noise
            noisy_real_images = add_instance_noise(real_images, noise_std)
            real_logits = discriminator(noisy_real_images)
            loss_d_real = criterion(real_logits, real_targets)

            # Fake images with small instance noise
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)

            noisy_fake_images = add_instance_noise(fake_images.detach(), noise_std)
            fake_logits = discriminator(noisy_fake_images)
            loss_d_fake = criterion(fake_logits, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # train generator

            generator.zero_grad(set_to_none=True)

            generator_targets = torch.full(
                (batch_size,),
                real_label_smooth,
                dtype=torch.float,
                device=device,
            )

            fake_logits_for_g = discriminator(fake_images)
            loss_g = criterion(fake_logits_for_g, generator_targets)

            loss_g.backward()
            optimizer_g.step()

            with torch.no_grad():
                d_real_prob = torch.sigmoid(real_logits).mean().item()
                d_fake_prob = torch.sigmoid(fake_logits).mean().item()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_d_real += d_real_prob
            epoch_d_fake += d_fake_prob

            progress_bar.set_postfix(
                {
                    "loss_d": f"{loss_d.item():.3f}",
                    "loss_g": f"{loss_g.item():.3f}",
                    "D(real)": f"{d_real_prob:.3f}",
                    "D(fake)": f"{d_fake_prob:.3f}",
                    "noise": f"{noise_std:.3f}",
                }
            )

        epoch_loss_d /= len(train_loader)
        epoch_loss_g /= len(train_loader)
        epoch_d_real /= len(train_loader)
        epoch_d_fake /= len(train_loader)

        log_row = {
            "epoch": epoch,
            "loss_d": epoch_loss_d,
            "loss_g": epoch_loss_g,
            "d_real": epoch_d_real,
            "d_fake": epoch_d_fake,
            "instance_noise_std": noise_std,
            "generator_lr": g_lr,
            "discriminator_lr": d_lr,
            "real_label_smooth": real_label_smooth,
        }

        training_log.append(log_row)

        print(
            f"Epoch {epoch}: "
            f"loss_d={epoch_loss_d:.4f}, "
            f"loss_g={epoch_loss_g:.4f}, "
            f"D(real)={epoch_d_real:.4f}, "
            f"D(fake)={epoch_d_fake:.4f}, "
            f"noise_std={noise_std:.4f}"
        )

        if epoch == 1 or epoch % config["training"]["save_every"] == 0 or epoch == num_epochs:
            sample_path = fixed_noise_dir / f"epoch_{epoch:03d}_generated.png"

            save_dcgan_samples(
                generator=generator,
                fixed_noise=fixed_noise,
                output_path=sample_path,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"dcgan_epoch_{epoch:03d}.pth",
            )

    final_checkpoint_path = project_root / dcgan_config["checkpoint_path"]
    ensure_dir(final_checkpoint_path.parent)

    torch.save(
        {
            "epoch": num_epochs,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "config": config,
        },
        final_checkpoint_path,
    )

    log_df = pd.DataFrame(training_log)
    log_path = runs_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"Saved final DCGAN checkpoint to: {final_checkpoint_path}")
    print(f"Saved DCGAN training log to: {log_path}")

    report_sample_path = project_root / config["reports"]["dcgan_samples_grid"]

    save_dcgan_samples(
        generator=generator,
        fixed_noise=fixed_noise,
        output_path=report_sample_path,
    )

    print(f"Saved DCGAN sample grid to: {report_sample_path}")

    return generator, discriminator, log_df