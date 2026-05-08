from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.utils import save_image

from src.data.transforms import resize_and_center_crop
from src.utils.paths import ensure_dir


SPLIT_MAP = {
    0: "train",
    1: "val",
    2: "test",
}


def find_image_dir(image_dir: str) -> Path:
    """
    Find the actual CelebA image directory.

    Some downloads have:
        img_align_celeba/000001.jpg

    Others have:
        img_align_celeba/img_align_celeba/000001.jpg

    This function handles both.
    """
    image_dir = Path(image_dir)

    if image_dir.exists() and any(image_dir.glob("*.jpg")):
        return image_dir

    nested_dir = image_dir / "img_align_celeba"
    if nested_dir.exists() and any(nested_dir.glob("*.jpg")):
        return nested_dir

    raise FileNotFoundError(
        f"Could not find CelebA images in {image_dir} or {nested_dir}"
    )


def read_partition_file(partition_file: str) -> pd.DataFrame:
    """
    Read CelebA partition CSV.

    Expected columns:
    - image_id
    - partition

    Partition values:
    - 0 = train
    - 1 = validation
    - 2 = test
    """
    partition_path = Path(partition_file)

    if not partition_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    df = pd.read_csv(partition_path)

    # Handle possible unnamed first column
    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})

    if "partition" not in df.columns:
        # If there are only two columns, assume second one is partition
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "partition"})
        else:
            raise ValueError("Could not find partition column in partition file.")

    df["split"] = df["partition"].map(SPLIT_MAP)

    return df[["image_id", "partition", "split"]]


def read_attribute_file(attr_file: str) -> pd.DataFrame:
    """
    Read CelebA attribute CSV.

    This is useful for later conditional generation.
    For now, we keep it in the metadata manifest.
    """
    attr_path = Path(attr_file)

    if not attr_path.exists():
        print(f"Attribute file not found, skipping: {attr_path}")
        return pd.DataFrame()

    df = pd.read_csv(attr_path)

    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})

    return df


def build_metadata(
    image_dir: Path,
    partition_df: pd.DataFrame,
    attr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build metadata DataFrame containing image paths and split info.
    """
    metadata = partition_df.copy()
    metadata["raw_path"] = metadata["image_id"].apply(lambda x: str(image_dir / x))

    if not attr_df.empty:
        metadata = metadata.merge(attr_df, on="image_id", how="left")

    metadata["exists"] = metadata["raw_path"].apply(lambda x: Path(x).exists())

    missing_count = (~metadata["exists"]).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} image files are missing.")

    metadata = metadata[metadata["exists"]].reset_index(drop=True)

    return metadata


def save_processed_images(
    metadata: pd.DataFrame,
    processed_dir: str,
    image_size: int = 64,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Resize and save processed images into train/val/test folders.
    """
    processed_dir = Path(processed_dir)

    for split in ["train", "val", "test"]:
        ensure_dir(processed_dir / split)

    if limit is not None:
        metadata = metadata.head(limit).copy()

    processed_paths = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing images"):
        raw_path = Path(row["raw_path"])
        split = row["split"]
        image_id = row["image_id"]

        output_path = processed_dir / split / image_id

        if not output_path.exists():
            try:
                image = Image.open(raw_path).convert("RGB")
                image = resize_and_center_crop(image, image_size=image_size)
                image.save(output_path, quality=95)
            except Exception as error:
                print(f"Failed to process {raw_path}: {error}")
                processed_paths.append(None)
                continue

        processed_paths.append(str(output_path))

    metadata = metadata.copy()
    metadata["processed_path"] = processed_paths
    metadata = metadata.dropna(subset=["processed_path"]).reset_index(drop=True)

    return metadata


def save_sample_grid(
    metadata: pd.DataFrame,
    output_path: str,
    image_size: int = 64,
    num_images: int = 64,
) -> None:
    """
    Save a small grid of real CelebA images for README/app/report use.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    sample_df = metadata[metadata["split"] == "train"].sample(
        n=min(num_images, len(metadata)),
        random_state=42,
    )

    to_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image_tensors = []

    for _, row in sample_df.iterrows():
        image_path = Path(row["processed_path"])

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensors.append(to_tensor(image))
        except Exception as error:
            print(f"Could not load image for sample grid: {image_path}, {error}")

    if len(image_tensors) == 0:
        raise ValueError("No images available to create sample grid.")

    grid = torch.stack(image_tensors)

    save_image(
        grid,
        output_path,
        nrow=8,
        normalize=False,
    )

    print(f"Saved sample grid to: {output_path}")


def prepare_celeba(config: Dict, limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full CelebA preparation pipeline.
    """
    data_config = config["data"]
    report_config = config["reports"]

    image_dir = find_image_dir(data_config["image_dir"])

    partition_df = read_partition_file(data_config["partition_file"])
    attr_df = read_attribute_file(data_config["attr_file"])

    metadata = build_metadata(
        image_dir=image_dir,
        partition_df=partition_df,
        attr_df=attr_df,
    )

    print("Dataset split counts:")
    print(metadata["split"].value_counts())

    processed_metadata = save_processed_images(
        metadata=metadata,
        processed_dir=data_config["processed_dir"],
        image_size=data_config["image_size"],
        limit=limit,
    )

    interim_dir = ensure_dir("data/interim")
    manifest_path = interim_dir / "celeba_manifest.csv"
    processed_metadata.to_csv(manifest_path, index=False)

    print(f"Saved manifest to: {manifest_path}")

    save_sample_grid(
        metadata=processed_metadata,
        output_path=report_config["real_samples_grid"],
        image_size=data_config["image_size"],
        num_images=64,
    )

    return metadata, processed_metadata