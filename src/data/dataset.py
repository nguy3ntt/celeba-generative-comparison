from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CelebAProcessedDataset(Dataset):
    """
    PyTorch Dataset for processed CelebA images.

    Expected manifest columns:
    - image_id
    - split
    - processed_path

    The processed_path is stored relative to the project root, e.g.
    data/processed/celeba_64/train/000001.jpg
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        project_root: str | Path = ".",
        limit: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform
        self.project_root = Path(project_root)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if limit is not None:
            self.df = self.df.head(limit).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["processed_path"])

        if not image_path.is_absolute():
            image_path = self.project_root / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Processed image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image