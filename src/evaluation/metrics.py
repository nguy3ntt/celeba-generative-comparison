from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.paths import ensure_dir


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    """
    Load CSV file if it exists.
    """
    path = Path(path)

    if not path.exists():
        print(f"Missing CSV: {path}")
        return None

    return pd.read_csv(path)


def build_model_summary(config: Dict, project_root: Path) -> pd.DataFrame:
    """
    Build a final model summary table using available training logs.
    """
    rows = []

    vae_log = load_csv_if_exists(project_root / "runs" / "vae" / "training_log.csv")

    if vae_log is not None:
        final = vae_log.iloc[-1]

        rows.append(
            {
                "model": "VAE",
                "epochs": int(final["epoch"]),
                "final_train_loss": final["train_total_loss"],
                "final_val_loss": final["val_total_loss"],
                "training_behaviour": "Stable",
                "visual_quality": "Low-Medium",
                "main_strength": "Stable latent-space baseline",
                "main_limitation": "Blurry generated outputs",
            }
        )

    dcgan_log = load_csv_if_exists(project_root / "runs" / "dcgan" / "training_log.csv")

    if dcgan_log is not None:
        final = dcgan_log.iloc[-1]

        rows.append(
            {
                "model": "Tuned DCGAN",
                "epochs": int(final["epoch"]),
                "final_train_loss": final["loss_g"],
                "final_val_loss": None,
                "training_behaviour": "Adversarial and less stable",
                "visual_quality": "Medium-High",
                "main_strength": "Sharper and more textured samples than VAE",
                "main_limitation": "GAN artifacts and late training instability",
            }
        )

    diffusion_log = load_csv_if_exists(project_root / "runs" / "diffusion" / "training_log.csv")

    if diffusion_log is not None:
        final = diffusion_log.iloc[-1]

        rows.append(
            {
                "model": "Diffusion",
                "epochs": int(final["epoch"]),
                "final_train_loss": final["train_loss"],
                "final_val_loss": None,
                "training_behaviour": "Stable",
                "visual_quality": "High",
                "main_strength": "Best visual quality and diversity",
                "main_limitation": "Slower sampling and heavier training",
            }
        )

    summary_df = pd.DataFrame(rows)

    output_path = project_root / "outputs" / "metrics" / "model_summary.csv"
    ensure_dir(output_path.parent)

    summary_df.to_csv(output_path, index=False)

    print(f"Saved model summary to: {output_path}")

    return summary_df