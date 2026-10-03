"""EduLens Streamlit entry point and shared dataset/settings router."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import streamlit as st

from src import config
from src.data_loader import DatasetBundle, load_uci_dataset, load_uploaded_csv
from src.ui.theme import load_theme
from src.validation import DataValidationError

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("edulens")
DatasetChoice = Literal["Portuguese (primary)", "Mathematics", "Upload CSV"]

st.set_page_config(
    page_title="EduLens | Student Performance Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_theme()


@st.cache_data(show_spinner="Loading and validating dataset…")
def _load_uci_cached(course: Literal["por", "mat"]) -> DatasetBundle:
    """Cache a validated local UCI dataset by course."""
    return load_uci_dataset(course)


@st.cache_data(show_spinner="Validating uploaded CSV…")
def _load_upload_cached(content_hash: str, content: bytes, file_name: str) -> DatasetBundle:
    """Cache an uploaded dataset by its content digest and display name."""
    del content_hash
    return load_uploaded_csv(content, file_name)


def _clear_dataset_cache() -> None:
    """Invalidate cached dataset results after the sidebar source changes."""
    _load_uci_cached.clear()
    _load_upload_cached.clear()
    st.session_state.pop("edulens_dataset_fingerprint", None)
    for key in (
        "el_explorer_search",
        "el_explorer_search_columns",
        "el_explorer_filter_column",
        "el_explorer_filter_values",
    ):
        st.session_state.pop(key, None)


def _settings_sidebar() -> dict[str, object]:
    """Render shared statistical settings and return the current session values."""
    with st.sidebar.expander("⚙ Settings", expanded=False):
        confidence = st.slider(
            "Confidence level",
            min_value=0.50,
            max_value=0.99,
            value=float(st.session_state.get("el_confidence", config.DEFAULT_CONFIDENCE)),
            step=0.01,
            format="%.2f",
            key="el_confidence",
            help="Confidence level used by inferential methods.",
        )
        alpha = st.slider(
            "Significance level (α)",
            min_value=0.01,
            max_value=0.20,
            value=float(st.session_state.get("el_alpha", config.DEFAULT_ALPHA)),
            step=0.01,
            format="%.2f",
            key="el_alpha",
            help="Reference threshold for statistical tests.",
        )
        low_upper = st.number_input(
            "Low performance: G3 <",
            min_value=0,
            max_value=19,
            value=int(
                st.session_state.get(
                    "el_low_upper",
                    config.PERFORMANCE_BANDS["LOW_UPPER_EXCLUSIVE"],
                )
            ),
            step=1,
            key="el_low_upper",
            help="The next grade begins the Medium band.",
        )
        medium_upper = st.number_input(
            "Medium performance upper bound (inclusive)",
            min_value=int(low_upper),
            max_value=19,
            value=max(
                int(low_upper),
                int(
                    st.session_state.get(
                        "el_medium_upper",
                        config.PERFORMANCE_BANDS["MEDIUM_UPPER_INCLUSIVE"],
                    )
                ),
            ),
            step=1,
            key="el_medium_upper",
            help="High performance starts at the next integer grade.",
        )
        high_lower = int(medium_upper) + 1
        st.caption(f"High performance: G3 ≥ {high_lower}")
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(st.session_state.get("el_seed", config.RANDOM_SEED)),
            step=1,
            key="el_seed",
            help="Used by reproducible model splits.",
        )
    return {
        "confidence_level": float(confidence),
        "alpha": float(alpha),
        "seed": int(seed),
        "thresholds": {
            "LOW_UPPER_EXCLUSIVE": int(low_upper),
            "MEDIUM_LOWER_INCLUSIVE": int(low_upper),
            "MEDIUM_UPPER_INCLUSIVE": int(medium_upper),
            "HIGH_LOWER_INCLUSIVE": high_lower,
        },
    }


def _load_selected_bundle(choice: DatasetChoice, uploaded_file: object) -> DatasetBundle | None:
    """Load a selected source through cached ingestion and report expected data errors kindly."""
    bundle: DatasetBundle | None = None
    fingerprint = choice
    try:
        if choice == "Portuguese (primary)":
            bundle = _load_uci_cached("por")
        elif choice == "Mathematics":
            bundle = _load_uci_cached("mat")
        elif uploaded_file is not None:
            content = uploaded_file.getvalue()
            file_name = uploaded_file.name
            digest = hashlib.sha256(content).hexdigest()
            fingerprint = f"upload:{digest}:{file_name}"
            bundle = _load_upload_cached(digest, content, file_name)
        else:
            st.info("Choose a CSV file to use the upload dataset.")
            return None
    except (DataValidationError, OSError) as exc:
        LOGGER.exception("Selected dataset could not be loaded")
        st.error(str(exc))
        return None
    previous = st.session_state.get("edulens_dataset_fingerprint")
    if previous is not None and previous != fingerprint:
        _load_uci_cached.clear()
        _load_upload_cached.clear()
        if choice == "Portuguese (primary)":
            bundle = _load_uci_cached("por")
        elif choice == "Mathematics":
            bundle = _load_uci_cached("mat")
        elif uploaded_file is not None:
            bundle = _load_upload_cached(digest, content, file_name)
    st.session_state["edulens_dataset_fingerprint"] = fingerprint
    return bundle


def _page_config(
    script: str,
    title: str,
    icon: str,
) -> st.Page:
    """Create a consistently labeled page from a views/ script."""
    return st.Page(
        str(Path(__file__).resolve().parent / "views" / script),
        title=title,
        icon=icon,
        default=script == "home.py",
    )


with st.sidebar:
    st.markdown(
        '<div class="el-brand" aria-label="EduLens">'
        '<span class="el-brand-mark" aria-hidden="true">EL</span>'
        '<span><span class="el-brand-name">EduLens</span><br>'
        '<span class="el-brand-tagline">Student performance intelligence</span></span>'
        "</div>",
        unsafe_allow_html=True,
    )
    dataset_choice: DatasetChoice = st.selectbox(
        "Dataset",
        options=("Portuguese (primary)", "Mathematics", "Upload CSV"),
        index=0,
        key="el_dataset_choice",
        on_change=_clear_dataset_cache,
        help="UCI courses are loaded separately; an uploaded CSV is held in memory.",
    )
    uploaded = None
    if dataset_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            accept_multiple_files=False,
            help="The CSV is validated and processed in memory; it is not saved to disk.",
            key="el_dataset_upload",
            on_change=_clear_dataset_cache,
        )

pages = {
    "Start here": [
        _page_config("home.py", "Home", "🏡"),
        _page_config("overview.py", "Overview", "🏠"),
        _page_config("dataset_explorer.py", "Dataset Explorer", "🗂️"),
    ],
    "Explore": [
        _page_config("descriptive_statistics.py", "Descriptive Statistics", "📊"),
        _page_config("exploratory_analysis.py", "Exploratory Analysis", "🔎"),
        _page_config("correlation.py", "Correlation", "↗️"),
        _page_config("probability.py", "Probability", "🎲"),
        _page_config("hypothesis_testing.py", "Hypothesis Testing", "🧪"),
    ],
    "Models": [
        _page_config("regression.py", "Regression", "📈"),
        _page_config("prediction.py", "Prediction", "🎯"),
        _page_config("model_evaluation.py", "Model Evaluation", "🧮"),
    ],
    "About": [_page_config("about.py", "About", "ℹ️")],
}

navigation = st.navigation(pages, position="hidden")
current_url = st.context.url
if isinstance(current_url, bytes):
    current_url = current_url.decode("utf-8")
elif not isinstance(current_url, str):
    current_url = ""
active_route = urlsplit(current_url).path.rstrip("/").rsplit("/", 1)[-1]
valid_routes = {
    Path(script).stem
    for script in (
        "overview.py",
        "dataset_explorer.py",
        "descriptive_statistics.py",
        "exploratory_analysis.py",
        "correlation.py",
        "probability.py",
        "hypothesis_testing.py",
        "regression.py",
        "prediction.py",
        "model_evaluation.py",
        "about.py",
    )
}
active_href = "" if active_route in {"", "overview"} else active_route
if active_href not in valid_routes and active_href != "":
    active_href = ""
with st.sidebar:
    with st.container(key="el-sidebar-navigation"):
        st.markdown(
            (
                "<style>"
                f'.st-key-el-sidebar-navigation a[data-testid="stPageLink-NavLink"]'
                f'[href="{active_href}"], '
                f'.st-key-el-sidebar-bottom a[data-testid="stPageLink-NavLink"]'
                f'[href="{active_href}"]'
                "{background:var(--gradient)!important;"
                "border-color:rgba(255,255,255,.18)!important;"
                "box-shadow:0 8px 24px rgba(148,93,151,.18),"
                "inset 0 1px 0 rgba(255,255,255,.16)!important;"
                "color:#fff!important;font-weight:700!important}"
                "</style>"
            ),
            unsafe_allow_html=True,
        )
        for section, section_pages in pages.items():
            if section == "About":
                continue
            st.markdown(
                f'<p class="el-sidebar-heading">{section}</p>',
                unsafe_allow_html=True,
            )
            for page in section_pages:
                st.page_link(page, label=page.title, icon=page.icon)

    with st.container(key="el-sidebar-bottom"):
        st.page_link(pages["About"][0], label="About", icon=pages["About"][0].icon)
        settings = _settings_sidebar()
        st.markdown(
            '<p class="el-source-note"><strong>Dataset citation</strong><br>'
            "Cortez (2008), <em>Student Performance</em>, UCI Machine Learning Repository.<br>"
            '<a href="https://doi.org/10.24432/C5TG7T">doi:10.24432/C5TG7T</a></p>',
            unsafe_allow_html=True,
        )

bundle = _load_selected_bundle(dataset_choice, uploaded)
st.session_state["edulens_dataset_bundle"] = bundle
st.session_state["edulens_settings"] = settings

navigation.run()
