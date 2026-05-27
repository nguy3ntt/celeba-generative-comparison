import streamlit as st

from app_utils import setup_import_path, show_image

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Project Overview",
    page_icon="📌",
    layout="wide",
)

st.title("Project Overview")

st.write(
    """
    This project compares three generative deep learning models for synthetic face generation
    using the CelebA dataset.
    """
)

st.subheader("Models Compared")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### VAE")
    st.write(
        """
        A Variational Autoencoder learns a compressed latent representation of face images.
        It is stable and interpretable, but usually produces blurry samples.
        """
    )

with col2:
    st.markdown("### Tuned DCGAN")
    st.write(
        """
        A DCGAN trains a generator and discriminator adversarially.
        It can produce sharper images than a VAE, but training is less stable.
        """
    )

with col3:
    st.markdown("### Diffusion")
    st.write(
        """
        A diffusion model learns to reverse a noising process.
        It starts from random noise and gradually denoises into a synthetic face.
        """
    )

st.divider()

st.subheader("End-to-End Pipeline")

st.code(
    """
CelebA dataset
    ↓
Preprocessing to 64x64 RGB images
    ↓
Train VAE, DCGAN, and Diffusion models
    ↓
Generate synthetic face samples
    ↓
Evaluate visual quality and training behaviour
    ↓
Run nearest-neighbour memorisation check
    ↓
Display results in Streamlit dashboard
    """,
    language="text",
)

st.subheader("Final Model Comparison")

comparison_path = PROJECT_ROOT / "reports" / "figures" / "model_comparison_grid.png"
show_image(comparison_path, caption="Generated sample comparison: VAE vs DCGAN vs Diffusion")