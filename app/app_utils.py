from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image


def get_project_root() -> Path:
    """
    Return the project root path
    """
    return Path(__file__).resolve().parents[1]


def setup_import_path() -> Path:
    """
    Add project root to sys.path so Streamlit pages can import from src/
    """
    project_root = get_project_root()

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    return project_root


def inject_custom_css():
    """
    Inject custom CSS to make the dashboard look more professional
    """
    st.markdown(
        """
        <style>
        /* Main layout */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0B0F19;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] div {
            color: #F5F5F5;
        }
        
        .sidebar-quote {
            padding: 0.95rem 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 14px;
            background: rgba(139, 92, 246, 0.12);
            border-left: 4px solid #8B5CF6;
            color: #D1D5DB;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .sidebar-quote-author {
            margin-top: 0.45rem;
            color: #A78BFA;
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* Hero card */
        .hero-card {
            padding: 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #1E1B4B 0%, #111827 45%, #0F172A 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.8rem;
            color: #FFFFFF;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #D1D5DB;
            max-width: 850px;
            line-height: 1.7;
        }

        .badge-row {
            margin-top: 1.2rem;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background-color: rgba(139, 92, 246, 0.18);
            color: #DDD6FE;
            border: 1px solid rgba(139, 92, 246, 0.35);
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Cards */
        .info-card {
            padding: 1.2rem 1.25rem;
            border-radius: 18px;
            background-color: #161B22;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            height: 100%;
        }

        .info-card h3 {
            font-size: 1.15rem;
            margin-bottom: 0.6rem;
            color: #FFFFFF;
        }

        .info-card p {
            color: #D1D5DB;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: #111827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.20);
        }

        .metric-label {
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            color: #FFFFFF;
            font-size: 1.35rem;
            font-weight: 800;
        }

        .metric-note {
            color: #A7F3D0;
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
        }

        .muted-text {
            color: #9CA3AF;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .small-divider {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        }

        /* Image styling */
        img {
            border-radius: 12px;
        }

        /* Hide Streamlit default footer */
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badges: list[str] | None = None):
    """
    Display a hero section
    """
    badge_html = ""

    if badges:
        badge_html = "<div class='badge-row'>"
        for badge in badges:
            badge_html += f"<span class='badge'>{badge}</span>"
        badge_html += "</div>"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str | None = None):
    note_html = f"<div class='metric-note'>{note}</div>" if note else ""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, description: str | None = None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

    if description:
        st.markdown(f"<div class='muted-text'>{description}</div>", unsafe_allow_html=True)


def divider():
    st.markdown("<div class='small-divider'></div>", unsafe_allow_html=True)


def load_image(path: str | Path):
    path = Path(path)

    if not path.exists():
        return None

    return Image.open(path).convert("RGB")


def show_image(
    path: str | Path,
    caption: str | None = None,
    use_container_width: bool = True,
):
    path = Path(path)

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=use_container_width,
        )
    else:
        st.warning(f"Image not found: `{path}`")


def sidebar_repo_link():
    """
    Display quote and GitHub repository link in the sidebar.
    """
    st.sidebar.markdown(
        """
        <div class="sidebar-quote">
            “An intelligent heart acquires knowledge, and the ear of the wise seeks knowledge.”
            <div class="sidebar-quote-author">— Proverbs 18:15</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Project Links")
    st.sidebar.markdown(
        "[GitHub Repository](https://github.com/nguy3ntt/celeba-generative-comparison)"
    )


def load_csv(path: str | Path):
    path = Path(path)

    if not path.exists():
        st.warning(f"CSV not found: `{path}`")
        return None

    return pd.read_csv(path)