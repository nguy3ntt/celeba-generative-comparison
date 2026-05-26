from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.data.dataset import CelebAProcessedDataset
from src.inference.generate import generate_diffusion_samples
from src.utils.paths import ensure_dir


def get_feature_transform(feature_size: int = 16):
    """
    Convert image to small flattened feature vector.

    This uses downsampled pixel features for a simple nearest-neighbour check.
    """
    return transforms.Compose(
        [
            transforms.Resize((feature_size, feature_size)),
            transforms.ToTensor(),
        ]
    )


def image_to_feature(image: Image.Image, transform) -> torch.Tensor:
    """
    Convert PIL image to flattened feature vector.
    """
    tensor = transform(image.convert("RGB"))
    return tensor.flatten()


def build_real_image_feature_bank(
    manifest_path: str | Path,
    project_root: Path,
    split: str = "train",
    max_real_images: int = 5000,
    feature_size: int = 16,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build feature vectors for a subset of real training images.
    """
    dataset = CelebAProcessedDataset(
        manifest_path=manifest_path,
        split=split,
        transform=None,
        project_root=project_root,
        limit=max_real_images,
    )

    transform = get_feature_transform(feature_size)

    features = []
    image_paths = []

    for i in tqdm(range(len(dataset)), desc="Building real image feature bank"):
        row = dataset.df.iloc[i]
        path = Path(row["processed_path"])

        if not path.is_absolute():
            path = project_root / path

        image = Image.open(path).convert("RGB")
        feature = image_to_feature(image, transform)

        features.append(feature)
        image_paths.append(str(path))

    feature_tensor = torch.stack(features)

    return feature_tensor, image_paths


def generated_tensor_to_features(
    generated_images: torch.Tensor,
    feature_size: int = 16,
) -> torch.Tensor:
    """
    Convert generated image tensors in [0, 1] into flattened feature vectors.
    """
    resize = transforms.Resize((feature_size, feature_size))

    features = []

    for image_tensor in generated_images:
        resized = resize(image_tensor)
        features.append(resized.flatten())

    return torch.stack(features)


def create_nearest_neighbor_grid(
    generated_images: torch.Tensor,
    nearest_image_paths: list[list[str]],
    output_path: str | Path,
    num_queries: int = 8,
) -> None:
    """
    Create a grid showing generated images beside their nearest real images.

    Each row:
    generated image | nearest real 1 | nearest real 2 | nearest real 3
    """
    rows = []
    to_tensor = transforms.ToTensor()

    for i in range(min(num_queries, generated_images.size(0))):
        row_images = []

        generated = generated_images[i]
        row_images.append(generated)

        for real_path in nearest_image_paths[i]:
            real_image = Image.open(real_path).convert("RGB")
            real_tensor = to_tensor(real_image)
            row_images.append(real_tensor)

        rows.extend(row_images)

    grid = make_grid(
        torch.stack(rows),
        nrow=1 + len(nearest_image_paths[0]),
    )

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    save_image(grid, output_path)

    print(f"Saved nearest-neighbour grid to: {output_path}")


def run_diffusion_nearest_neighbor_check(
    config: Dict,
    project_root: Path,
    num_generated: int = 8,
    num_neighbors: int = 3,
    max_real_images: int = 5000,
    feature_size: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate diffusion samples and compare each sample with nearest real training images.
    """
    manifest_path = project_root / "data" / "interim" / "celeba_manifest.csv"

    real_features, real_paths = build_real_image_feature_bank(
        manifest_path=manifest_path,
        project_root=project_root,
        split="train",
        max_real_images=max_real_images,
        feature_size=feature_size,
    )

    generated_images = generate_diffusion_samples(
        config=config,
        project_root=project_root,
        num_images=num_generated,
        seed=seed,
        inference_steps=config["diffusion"]["inference_steps"],
    )

    generated_features = generated_tensor_to_features(
        generated_images=generated_images,
        feature_size=feature_size,
    )

    nn_model = NearestNeighbors(
        n_neighbors=num_neighbors,
        metric="euclidean",
    )

    nn_model.fit(real_features.numpy())

    distances, indices = nn_model.kneighbors(generated_features.numpy())

    nearest_paths = []

    rows = []

    for query_idx in range(num_generated):
        query_paths = []

        for rank in range(num_neighbors):
            real_idx = indices[query_idx][rank]
            distance = distances[query_idx][rank]
            real_path = real_paths[real_idx]

            query_paths.append(real_path)

            rows.append(
                {
                    "generated_index": query_idx,
                    "neighbor_rank": rank + 1,
                    "real_image_path": real_path,
                    "distance": distance,
                }
            )

        nearest_paths.append(query_paths)

    output_grid_path = project_root / config["reports"]["nearest_neighbor_examples"]

    create_nearest_neighbor_grid(
        generated_images=generated_images,
        nearest_image_paths=nearest_paths,
        output_path=output_grid_path,
        num_queries=num_generated,
    )

    result_df = pd.DataFrame(rows)

    output_csv_path = project_root / "outputs" / "metrics" / "nearest_neighbor_distances.csv"
    ensure_dir(output_csv_path.parent)

    result_df.to_csv(output_csv_path, index=False)

    print(f"Saved nearest-neighbour distances to: {output_csv_path}")

    return result_df