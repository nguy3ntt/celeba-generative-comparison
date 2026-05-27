import streamlit as st

from app_utils import (
    setup_import_path,
    inject_custom_css,
    hero,
    info_card,
    metric_card,
    section_title,
    divider,
    show_image,
    sidebar_repo_link,
)

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="CelebA Generative Model Comparison",
    page_icon="🧠",
    layout="wide",
)

inject_custom_css()
sidebar_repo_link()

hero(
    title="CelebA Generative Model Comparison",
    subtitle=(
        "An end-to-end generative deep learning dashboard comparing VAE, tuned DCGAN, "
        "and DDPM-style diffusion models for synthetic face generation on CelebA."
    ),
    badges=[
        "PyTorch",
        "CelebA",
        "VAE",
        "DCGAN",
        "Diffusion",
        "Streamlit",
        "Responsible AI",
    ],
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Dataset", "CelebA (Kaggle)", "64×64 RGB processed images")

with col2:
    metric_card("Models", "3", "VAE, DCGAN, Diffusion")

with col3:
    metric_card("Training Budget", "30 epochs", "same epoch budget per model")

with col4:
    metric_card("Best Model", "Diffusion", "strongest visual quality")

divider()

section_title(
    "Project Summary",
    "This project compares three major generative model families using the same processed CelebA dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    info_card(
        "VAE Baseline",
        "The VAE provides a stable latent-space baseline. It learns broad face structure, but generated outputs are smooth and blurry."
    )

with col2:
    info_card(
        "Tuned DCGAN",
        "The DCGAN produces sharper and more textured faces than the VAE, but adversarial training introduces instability and artifacts."
    )

with col3:
    info_card(
        "Diffusion Model",
        "The diffusion model produces the strongest visual results, with clearer facial structure, better diversity, and fewer obvious artifacts."
    )

divider()

section_title("Final Generated Sample Comparison")

comparison_path = PROJECT_ROOT / "reports" / "figures" / "model_comparison_grid.png"
show_image(
    comparison_path,
    caption="Generated sample comparison: VAE vs DCGAN vs Diffusion",
)

divider()

section_title("End-to-End Pipeline")

st.code(
    """
CelebA dataset
    ↓
Preprocessing to 64×64 RGB images
    ↓
Train VAE, tuned DCGAN, and diffusion models
    ↓
Generate synthetic face samples
    ↓
Compare training behaviour and visual quality
    ↓
Run nearest-neighbour memorisation check
    ↓
Present results in Streamlit dashboard
    """,
    language="text",
)

st.info(
    """
    This app currently runs in lightweight demo mode: it displays saved model outputs and evaluation results.
    This makes the dashboard suitable for GitHub and public deployment without requiring GPU inference.
    """
)