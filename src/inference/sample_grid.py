from pathlib import Path

import torch
from torchvision.utils import save_image

from src.utils.paths import ensure_dir


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensors from [-1, 1] to [0, 1].
    """
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def save_tensor_grid(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    normalize: bool = False,
    value_range: tuple[float, float] | None = None,
) -> None:
    """
    Save a tensor image grid to disk.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    save_image(
        images,
        output_path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
    )

    print(f"Saved image grid to: {output_path}")