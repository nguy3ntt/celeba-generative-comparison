import streamlit as st
import pandas as pd

from app_utils import setup_import_path, show_image, load_csv

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="🖼️",
    layout="wide",
)

st.title("Dataset Explorer")

st.write(
    """
    This page summarises the CelebA dataset preparation stage.
    The full dataset is kept locally and is not committed to GitHub.
    """
)

st.subheader("Dataset Used")

st.markdown(
    """
    **CelebA** is a large-scale face image dataset containing aligned face images and metadata such as facial attributes,
    bounding boxes, landmarks, and train/validation/test split information.

    In this project, images are processed into **64x64 RGB** format for VAE, DCGAN, and diffusion training.
    """
)

st.subheader("Real CelebA Samples")

real_samples_path = PROJECT_ROOT / "reports" / "figures" / "real_samples_grid.png"
show_image(real_samples_path, caption="Real processed CelebA training samples")

st.subheader("Split Distribution")

split_plot_path = PROJECT_ROOT / "reports" / "figures" / "split_distribution.png"
show_image(split_plot_path, caption="Train / validation / test split distribution")

st.subheader("Attribute Distribution")

attr_plot_path = PROJECT_ROOT / "reports" / "figures" / "top_attribute_distribution.png"
show_image(attr_plot_path, caption="Top CelebA attributes by positive rate")

st.subheader("Local Dataset Structure")

st.code(
    """
data/
  raw/
    celeba/
      img_align_celeba/
      list_attr_celeba.csv
      list_bbox_celeba.csv
      list_eval_partition.csv
      list_landmarks_align_celeba.csv

  processed/
    celeba_64/
      train/
      val/
      test/
    """,
    language="text",
)

st.info(
    """
    The raw and processed dataset folders are local-only and excluded from GitHub using `.gitignore`.
    """
)