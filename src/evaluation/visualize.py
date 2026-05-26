from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from PIL import Image

from src.utils.paths import ensure_dir


def create_model_comparison_figure(
    image_paths: Dict[str, str | Path],
    output_path: str | Path,
) -> None:
    """
    Create a side-by-side comparison figure for generated sample grids.
    """
    available_images = {}

    for model_name, path in image_paths.items():
        path = Path(path)

        if path.exists():
            available_images[model_name] = Image.open(path).convert("RGB")
        else:
            print(f"Missing image for {model_name}: {path}")

    if len(available_images) == 0:
        raise ValueError("No images found for comparison figure.")

    fig, axes = plt.subplots(
        1,
        len(available_images),
        figsize=(6 * len(available_images), 6),
    )

    if len(available_images) == 1:
        axes = [axes]

    for ax, (model_name, image) in zip(axes, available_images.items()):
        ax.imshow(image)
        ax.set_title(model_name, fontsize=14)
        ax.axis("off")

    plt.suptitle("Generated Sample Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved comparison figure to: {output_path}")