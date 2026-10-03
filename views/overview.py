"""Overview page for the selected student performance dataset."""

from __future__ import annotations

import streamlit as st

from src.data_loader import DatasetBundle
from src.preprocessing import derive_performance_category
from src.ui.components import metric_card, pipeline_strip, status_chip

st.markdown(
    '<section class="el-hero"><h1>Understand What Shapes Student Performance</h1>'
    "<p>Explore student performance data with transparent descriptive statistics, "
    "probability, and carefully checked models.</p></section>",
    unsafe_allow_html=True,
)
selected = st.session_state.get("edulens_dataset_bundle")
bundle = selected if isinstance(selected, DatasetBundle) else None

if bundle is None:
    st.info("Select a course dataset or upload a CSV from the sidebar to get started.")
else:
    st.caption(
        f"Current dataset: {bundle.metadata.file_name} · "
        f"{bundle.metadata.source} · loaded as a separate course dataset."
    )
    if "G3" in bundle.clean.columns:
        settings = st.session_state.get("edulens_settings", {})
        thresholds = settings.get("thresholds") if isinstance(settings, dict) else None
        categories = derive_performance_category(bundle.clean, thresholds=thresholds)
        columns = st.columns(4)
        with columns[0]:
            metric_card("Student records", f"{bundle.clean.shape[0]:,}")
        with columns[1]:
            metric_card("Variables", f"{bundle.metadata.column_count:,}")
        with columns[2]:
            metric_card("Missing source values", f"{bundle.cleaning_report.missing_cells:,}")
        with columns[3]:
            metric_card("Unclassified grades", f"{categories.unclassified_count:,}")
        st.subheader("Performance bands")
        st.caption(
            "Labels use the current sidebar grade thresholds. Counts are computed from the "
            "selected dataset; labels describe grade bands, not student traits."
        )
        band_columns = st.columns(3)
        for column, label in zip(band_columns, ("Low", "Medium", "High"), strict=True):
            with column:
                metric_card(f"{label} performance", f"{categories.class_counts[label]:,}")
        status_chip("Bands computed from the selected dataset", "info")
    else:
        columns = st.columns(3)
        with columns[0]:
            metric_card("Student records", f"{bundle.clean.shape[0]:,}")
        with columns[1]:
            metric_card("Variables", f"{bundle.metadata.column_count:,}")
        with columns[2]:
            metric_card("Missing source values", f"{bundle.cleaning_report.missing_cells:,}")
        status_chip("G3 is not present; performance bands are unavailable", "notice")

st.subheader("How EduLens Works")
st.markdown(
    '<p class="el-caption">The workflow keeps data inspection, statistical analysis, and '
    "modeling distinct. Results are calculated from the selected dataset and shown with "
    "assumptions and limitations.</p>",
    unsafe_allow_html=True,
)
pipeline_strip()
