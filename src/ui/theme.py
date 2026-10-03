"""Theme loading and one-time CSS injection for the EduLens Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

THEME_TOKENS = {
    "background": "#111014",
    "surface": "#19171D",
    "card": "#211E25",
    "border": "rgba(245, 230, 248, 0.12)",
    "primary": "#D9B4F2",
    "secondary": "#E2A6C7",
    "text": "#F5F1F5",
    "muted": "#B9AFBB",
    "gradient": "linear-gradient(135deg, #6F5278 0%, #855F7F 54%, #604762 100%)",
    "shadow": "0 18px 50px rgba(0, 0, 0, 0.36)",
    "radius": "22px",
}


def load_theme() -> None:
    """Inject the repository stylesheet once per Streamlit run."""
    css_path = Path(__file__).resolve().parents[2] / "assets" / "styles" / "edulens.css"
    try:
        stylesheet = css_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"EduLens stylesheet could not be loaded: {css_path}") from exc
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
