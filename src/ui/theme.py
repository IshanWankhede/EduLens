"""Theme loading and one-time CSS injection for the EduLens Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def load_theme() -> None:
    """Inject the repository stylesheet once per Streamlit run."""
    css_path = Path(__file__).resolve().parents[2] / "assets" / "styles" / "edulens.css"
    try:
        stylesheet = css_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"EduLens stylesheet could not be loaded: {css_path}") from exc
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
