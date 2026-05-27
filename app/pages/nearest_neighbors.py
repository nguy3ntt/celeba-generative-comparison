import streamlit as st

from app_utils import setup_import_path, load_csv, show_image

PROJECT_ROOT = setup_import_path()

st.set_page_config(
    page_title="Nearest Neighbours",
    page_icon="🔎",
    layout="wide",
)

st.title("Nearest-Neighbour Check")

st.write(
    """
    This page checks whether generated diffusion samples appear too similar to real training images.
    This is especially important because CelebA contains real human face images.
    """
)

nearest_grid_path = PROJECT_ROOT / "reports" / "figures" / "nearest_neighbour_examples.png"
nearest_csv_path = PROJECT_ROOT / "outputs" / "metrics" / "nearest_neighbor_distances.csv"

st.subheader("Generated Images and Closest Real Training Images")

show_image(
    nearest_grid_path,
    caption="Each row shows a generated diffusion sample followed by its nearest real training images.",
)

st.subheader("Nearest-Neighbour Distances")

nn_df = load_csv(nearest_csv_path)

if nn_df is not None:
    st.dataframe(nn_df.head(50), use_container_width=True)

    st.write("Minimum distance per generated image:")

    min_dist = nn_df.groupby("generated_index")["distance"].min().reset_index()
    st.dataframe(min_dist, use_container_width=True)

st.subheader("Interpretation")

st.write(
    """
    The nearest-neighbour check is a qualitative memorisation check. It compares generated images with visually similar
    real training images using simple downsampled pixel-space features.

    This does not prove identity-level similarity, but it helps inspect whether the model appears to copy exact training
    samples or instead generates new synthetic faces that broadly match the CelebA distribution.
    """
)

st.warning(
    """
    Because this project uses real face data, generated samples should be treated as synthetic educational outputs only.
    They should not be used for impersonation or misleading media.
    """
)