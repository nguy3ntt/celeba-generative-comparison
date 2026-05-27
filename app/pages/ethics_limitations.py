import streamlit as st

from app_utils import setup_import_path

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Ethics & Limitations",
    page_icon="⚠️",
    layout="wide",
)

st.title("Ethics & Limitations")

st.subheader("Responsible Use")

st.write(
    """
    This project is for educational generative modelling only.
    The goal is to compare VAE, DCGAN, and diffusion models, not to generate or imitate any specific real person.
    """
)

st.markdown(
    """
    The generated outputs should be treated as **synthetic samples** and should not be used for:

    - impersonation
    - identity manipulation
    - misleading media
    - misinformation
    - face-based profiling
    """
)

st.subheader("Dataset Limitation")

st.write(
    """
    CelebA contains real human face images. This makes the dataset useful for studying generative modelling,
    but it also means the project needs responsible presentation and memorisation checks.
    """
)

st.subheader("Model Limitations")

st.markdown(
    """
    **VAE limitations**
    - Produces blurry outputs
    - Loses fine details during reconstruction and sampling

    **DCGAN limitations**
    - Training is unstable
    - Can suffer from mode collapse
    - Generated faces may contain artifacts or distortions

    **Diffusion limitations**
    - Slower sampling
    - Heavier training cost
    - Strong outputs require memorisation checks
    """
)

st.subheader("Evaluation Limitations")

st.write(
    """
    The nearest-neighbour check in this project is a useful first inspection, but it is not a complete identity-level
    memorisation audit. More advanced checks could use perceptual embeddings or face-recognition-style features.
    """
)

st.subheader("Future Work")

st.markdown(
    """
    Possible future improvements:

    - Add FID/KID quantitative metrics
    - Add attribute-conditioned generation using CelebA labels
    - Add live local inference mode
    - Deploy lightweight public dashboard
    - Use perceptual nearest-neighbour checks instead of only pixel-space features
    """
)