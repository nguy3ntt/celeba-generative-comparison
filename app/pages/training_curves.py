import streamlit as st
import matplotlib.pyplot as plt

from app_utils import setup_import_path, load_csv, show_image

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Training Curves",
    page_icon="📈",
    layout="wide",
)

st.title("Training Curves")

st.write(
    """
    This page compares the training behaviour of the VAE, tuned DCGAN, and diffusion model.
    """
)

vae_log_path = PROJECT_ROOT / "runs" / "vae" / "training_log.csv"
dcgan_log_path = PROJECT_ROOT / "runs" / "dcgan" / "training_log.csv"
diffusion_log_path = PROJECT_ROOT / "runs" / "diffusion" / "training_log.csv"

tab1, tab2, tab3 = st.tabs(["VAE", "Tuned DCGAN", "Diffusion"])

with tab1:
    st.subheader("VAE Training Behaviour")

    vae_log = load_csv(vae_log_path)

    if vae_log is not None:
        st.dataframe(vae_log.tail())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(vae_log["epoch"], vae_log["train_total_loss"], marker="o", label="Train Total Loss")
        ax.plot(vae_log["epoch"], vae_log["val_total_loss"], marker="o", label="Validation Total Loss")
        ax.set_title("VAE Total Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.write(
            """
            The VAE trained smoothly, with both training and validation losses decreasing steadily.
            This reflects the stability of reconstruction-based generative learning.
            """
        )

with tab2:
    st.subheader("Tuned DCGAN Training Behaviour")

    dcgan_log = load_csv(dcgan_log_path)

    if dcgan_log is not None:
        st.dataframe(dcgan_log.tail())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dcgan_log["epoch"], dcgan_log["loss_d"], marker="o", label="Discriminator Loss")
        ax.plot(dcgan_log["epoch"], dcgan_log["loss_g"], marker="o", label="Generator Loss")
        ax.set_title("DCGAN Generator and Discriminator Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dcgan_log["epoch"], dcgan_log["d_real"], marker="o", label="D(real)")
        ax.plot(dcgan_log["epoch"], dcgan_log["d_fake"], marker="o", label="D(fake)")
        ax.set_title("DCGAN Discriminator Confidence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.write(
            """
            The DCGAN showed less stable training because the generator and discriminator compete against each other.
            The tuned version improved sample diversity, but late training instability still appeared.
            """
        )

with tab3:
    st.subheader("Diffusion Training Behaviour")

    diffusion_log = load_csv(diffusion_log_path)

    if diffusion_log is not None:
        st.dataframe(diffusion_log.tail())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(diffusion_log["epoch"], diffusion_log["train_loss"], marker="o", label="Noise Prediction Loss")
        ax.set_title("Diffusion Noise Prediction Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.write(
            """
            The diffusion model trained stably. Its noise-prediction loss decreased early and then stabilised,
            suggesting that the U-Net learned the denoising task effectively.
            """
        )