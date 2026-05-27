import streamlit as st

from app_utils import (
    setup_import_path,
    inject_custom_css,
    hero,
    metric_card,
    section_title,
    show_image,
    load_csv,
    divider,
)

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Evaluation Metrics",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()

hero(
    title="Evaluation Metrics",
    subtitle=(
        "A model-level summary of training behaviour, visual quality, strengths, limitations, "
        "and generated sample comparison."
    ),
    badges=["Model summary", "Training behaviour", "Visual comparison"],
)

model_summary_path = PROJECT_ROOT / "reports" / "tables" / "model_summary.csv"
comparison_grid_path = PROJECT_ROOT / "reports" / "figures" / "model_comparison_grid.png"

col1, col2, col3 = st.columns(3)

with col1:
    metric_card("VAE", "Stable", "blurry generated outputs")

with col2:
    metric_card("Tuned DCGAN", "Sharper", "visible GAN artifacts")

with col3:
    metric_card("Diffusion", "Best overall", "highest visual quality")

divider()

section_title("Model Summary Table")

model_summary = load_csv(model_summary_path)

if model_summary is not None:
    st.dataframe(model_summary, use_container_width=True)

divider()

section_title("Generated Sample Comparison")

show_image(
    comparison_grid_path,
    caption="Final generated sample comparison",
)

divider()

section_title("Interpretation")

st.markdown(
    """
    | Model | Main Result |
    |---|---|
    | **VAE** | Stable training, but generated outputs are blurry |
    | **Tuned DCGAN** | Sharper and more textured outputs, but less stable |
    | **Diffusion** | Best visual quality, diversity, and stability |
    """
)

st.info(
    """
    FID/KID-style metrics can be added later as an extension. The current version focuses on visual comparison,
    training behaviour, and nearest-neighbour checking.
    """
)