from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import CelebAProcessedDataset
from src.models.vae import ConvVAE
from src.utils.paths import ensure_dir


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1, 1] back to [0, 1] for saving/display.
    """
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def vae_loss_function(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = reconstruction loss + beta * KL divergence.
    """
    recon_loss = F.mse_loss(reconstructed, original, reduction="sum") / original.size(0)

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / original.size(0)

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def get_vae_transforms(config: Dict):
    """
    Transform processed PIL images into normalized tensors.
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


def create_dataloaders(
    config: Dict,
    project_root: Path,
    limit: Optional[int] = None,
):
    """
    Create train and validation dataloaders.
    """
    transform = get_vae_transforms(config)

    manifest_path = project_root / "data" / "interim" / "celeba_manifest.csv"

    train_dataset = CelebAProcessedDataset(
        manifest_path=manifest_path,
        split="train",
        transform=transform,
        project_root=project_root,
        limit=limit,
    )

    val_dataset = CelebAProcessedDataset(
        manifest_path=manifest_path,
        split="val",
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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
    )

    return train_loader, val_loader


def save_vae_outputs(
    model: ConvVAE,
    images: torch.Tensor,
    device: torch.device,
    epoch: int,
    output_dirs: Dict,
    num_samples: int = 64,
):
    """
    Save reconstruction and generated sample grids.
    """
    model.eval()

    recon_dir = ensure_dir(output_dirs["reconstructions_dir"])
    generated_dir = ensure_dir(output_dirs["generated_dir"])

    with torch.no_grad():
        images = images.to(device)
        reconstructed, _, _ = model(images)

        comparison = torch.cat(
            [
                images[:8],
                reconstructed[:8],
            ],
            dim=0,
        )

        comparison = denormalize_images(comparison)

        save_image(
            comparison,
            recon_dir / f"epoch_{epoch:03d}_reconstructions.png",
            nrow=8,
        )

        samples = model.sample(num_samples=num_samples, device=device)
        samples = denormalize_images(samples)

        save_image(
            samples,
            generated_dir / f"epoch_{epoch:03d}_generated.png",
            nrow=8,
        )


def train_vae(
    config: Dict,
    project_root: Path,
    epochs: Optional[int] = None,
    limit: Optional[int] = None,
):
    """
    Full VAE training loop.
    """
    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        config=config,
        project_root=project_root,
        limit=limit,
    )

    vae_config = config["vae"]
    data_config = config["data"]

    model = ConvVAE(
        image_channels=data_config["channels"],
        latent_dim=vae_config["latent_dim"],
        hidden_dims=vae_config["hidden_dims"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=vae_config["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    beta = vae_config["beta"]
    num_epochs = epochs if epochs is not None else vae_config["epochs"]

    checkpoint_dir = ensure_dir(project_root / "checkpoints" / "vae")
    runs_dir = ensure_dir(project_root / "runs" / "vae")

    output_dirs = {
        "reconstructions_dir": project_root / vae_config["outputs"]["reconstructions_dir"],
        "generated_dir": project_root / vae_config["outputs"]["generated_dir"],
    }

    training_log = []

    fixed_batch = next(iter(train_loader))

    for epoch in range(1, num_epochs + 1):
        model.train()

        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=True,
        )

        for images in progress_bar:
            images = images.to(device)

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(images)

            loss, recon_loss, kl_loss = vae_loss_function(
                reconstructed=reconstructed,
                original=images,
                mu=mu,
                logvar=logvar,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.2f}",
                    "recon": f"{recon_loss.item():.2f}",
                    "kl": f"{kl_loss.item():.2f}",
                }
            )

        train_total_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)

                reconstructed, mu, logvar = model(images)

                loss, recon_loss, kl_loss = vae_loss_function(
                    reconstructed=reconstructed,
                    original=images,
                    mu=mu,
                    logvar=logvar,
                    beta=beta,
                )

                val_total_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

        val_total_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)

        log_row = {
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_recon_loss": train_recon_loss,
            "train_kl_loss": train_kl_loss,
            "val_total_loss": val_total_loss,
            "val_recon_loss": val_recon_loss,
            "val_kl_loss": val_kl_loss,
        }

        training_log.append(log_row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_total_loss:.2f}, "
            f"val_loss={val_total_loss:.2f}"
        )

        if epoch == 1 or epoch % config["training"]["save_every"] == 0 or epoch == num_epochs:
            save_vae_outputs(
                model=model,
                images=fixed_batch,
                device=device,
                epoch=epoch,
                output_dirs=output_dirs,
                num_samples=64,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"vae_epoch_{epoch:03d}.pth",
            )

    # save final checkpoint
    final_checkpoint_path = project_root / vae_config["checkpoint_path"]
    ensure_dir(final_checkpoint_path.parent)

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        final_checkpoint_path,
    )

    # save training log
    log_df = pd.DataFrame(training_log)
    log_path = runs_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"Saved final VAE checkpoint to: {final_checkpoint_path}")
    print(f"Saved VAE training log to: {log_path}")

    # save final generated sample grid for public report figure
    reports_dir = ensure_dir(project_root / config["reports"]["figures_dir"])
    model.eval()

    with torch.no_grad():
        samples = model.sample(num_samples=64, device=device)
        samples = denormalize_images(samples)

        report_sample_path = reports_dir / "vae_samples_grid.png"

        save_image(
            samples,
            report_sample_path,
            nrow=8,
        )

    print(f"Saved VAE sample grid to: {report_sample_path}")

    return model, log_df