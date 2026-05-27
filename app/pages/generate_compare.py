import streamlit as st

from app_utils import (
    setup_import_path,
    inject_custom_css,
    hero,
    section_title,
    info_card,
    show_image,
    divider,
)

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Generate & Compare",
    page_icon="🎨",
    layout="wide",
)

inject_custom_css()

hero(
    title="Generate & Compare",
    subtitle=(
        "Compare saved generated samples from the VAE, tuned DCGAN, and diffusion models. "
        "This page highlights the visual trade-offs between stability, sharpness, and diversity."
    ),
    badges=["Lightweight demo mode", "Saved generated outputs", "Model comparison"],
)

vae_path = PROJECT_ROOT / "reports" / "figures" / "vae_samples_grid.png"
dcgan_path = PROJECT_ROOT / "reports" / "figures" / "dcgan_samples_grid.png"
diffusion_path = PROJECT_ROOT / "reports" / "figures" / "diffusion_samples_grid.png"

selected_model = st.selectbox(
    "Select model view",
    ["All Models", "VAE", "Tuned DCGAN", "Diffusion"],
)

divider()

if selected_model == "All Models":
    section_title("Generated Samples Across Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("VAE")
        show_image(vae_path, caption="Stable but blurry")
        info_card(
            "VAE Result",
            "The VAE generates smooth face-like samples, but fine details are lost because the model learns through reconstruction and latent regularisation."
        )

    with col2:
        st.subheader("Tuned DCGAN")
        show_image(dcgan_path, caption="Sharper but unstable")
        info_card(
            "DCGAN Result",
            "The tuned DCGAN creates sharper textures than the VAE, but some generated images contain visual artifacts from adversarial training."
        )

    with col3:
        st.subheader("Diffusion")
        show_image(diffusion_path, caption="Best visual quality")
        info_card(
            "Diffusion Result",
            "The diffusion model produces the strongest samples overall, with clearer structure, more realistic colours, and stronger diversity."
        )

elif selected_model == "VAE":
    section_title("VAE Generated Samples")
    show_image(vae_path, caption="VAE generated samples")
    info_card(
        "Interpretation",
        "The VAE is a stable generative baseline. It learns the broad face distribution, but samples are blurry and lack high-frequency details."
    )

elif selected_model == "Tuned DCGAN":
    section_title("Tuned DCGAN Generated Samples")
    show_image(dcgan_path, caption="DCGAN generated samples")
    info_card(
        "Interpretation",
        "The DCGAN improves visual sharpness compared with VAE. However, it is less stable and may produce artifacts or distorted facial regions."
    )

elif selected_model == "Diffusion":
    section_title("Diffusion Generated Samples")
    show_image(diffusion_path, caption="Diffusion generated samples")
    info_card(
        "Interpretation",
        "The diffusion model gives the best result in this project. It generates more realistic, diverse, and visually coherent synthetic faces."
    )

divider()

section_title("Main Finding")

st.success(
    """
    The VAE is stable but blurry, the tuned DCGAN is sharper but unstable, and the diffusion model gives the best balance of visual quality and diversity.
    """
)