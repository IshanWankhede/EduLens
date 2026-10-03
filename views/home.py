"""Premium landing page for EduLens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DatasetBundle
from src.preprocessing import derive_performance_category
from src.ui.components import metric_card


def _page(script: str, title: str, icon: str) -> st.Page:
    """Create a link target for one of the existing routed view scripts."""
    return st.Page(str(Path(__file__).resolve().with_name(script)), title=title, icon=icon)


pages = {
    "overview": _page("overview.py", "Overview", "🏠"),
    "descriptive": _page("descriptive_statistics.py", "Descriptive Statistics", "📊"),
    "probability": _page("probability.py", "Probability", "🎲"),
    "hypothesis": _page("hypothesis_testing.py", "Hypothesis Testing", "🧪"),
    "regression": _page("regression.py", "Regression", "📈"),
    "prediction": _page("prediction.py", "Prediction", "🎯"),
    "explorer": _page("dataset_explorer.py", "Dataset Explorer", "🗂️"),
}

st.markdown(
    """
    <section class="el-home-hero">
      <span class="el-home-eyebrow">EDULENS · STUDENT PERFORMANCE STATISTICS</span>
      <h1>Understand What Shapes Student Performance</h1>
      <p class="el-home-tagline">
        Discover the factors. Understand the patterns. Predict the probability.
      </p>
      <p class="el-home-intro">
        Explore real student-performance data with transparent statistics, careful
        interpretation, and reproducible models.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(key="home-hero-actions"):
    action_columns = st.columns(2)
    with action_columns[0]:
        st.page_link(
            pages["overview"],
            label="Explore the Dashboard",
            icon="🔎",
            use_container_width=True,
        )
    with action_columns[1]:
        st.page_link(
            pages["descriptive"],
            label="See the Statistics",
            icon="📊",
            use_container_width=True,
        )

st.markdown(
    '<div class="el-home-section-heading"><div><span class="el-home-eyebrow">'
    "SELECTED DATASET</span><h2>At a glance</h2></div>"
    '<span class="el-home-live-indicator"><span aria-hidden="true">●</span> '
    "Computed from the active data</span></div>",
    unsafe_allow_html=True,
)

selected = st.session_state.get("edulens_dataset_bundle")
bundle = selected if isinstance(selected, DatasetBundle) else None

if bundle is None:
    st.info("Select a valid course dataset or upload a CSV from the sidebar to see its summary.")
else:
    metric_columns = st.columns(4)
    with metric_columns[0]:
        metric_card("Students", f"{len(bundle.clean):,}")
    with metric_columns[1]:
        metric_card("Variables", f"{bundle.metadata.column_count:,}")

    if "G3" in bundle.clean.columns and pd.api.types.is_numeric_dtype(bundle.clean["G3"]):
        valid_grades = bundle.clean["G3"].dropna()
        if not valid_grades.empty:
            with metric_columns[2]:
                metric_card("Mean final grade · G3", f"{valid_grades.mean():.2f}")
            settings = st.session_state.get("edulens_settings", {})
            thresholds = settings.get("thresholds") if isinstance(settings, dict) else None
            categories = derive_performance_category(bundle.clean, thresholds=thresholds)
            classifiable_count = sum(categories.class_counts.values())
            if classifiable_count:
                high_percent = categories.class_counts["High"] / classifiable_count * 100
                with metric_columns[3]:
                    metric_card("High performance", f"{high_percent:.1f}%")
            else:
                with metric_columns[3]:
                    metric_card("High performance", "Unavailable", "No classifiable G3 values.")
        else:
            with metric_columns[2]:
                metric_card("Mean final grade · G3", "Unavailable", "No valid G3 values.")
            with metric_columns[3]:
                metric_card("High performance", "Unavailable", "No valid G3 values.")
    else:
        with metric_columns[2]:
            metric_card(
                "Mean final grade · G3", "Unavailable", "G3 is not present in this dataset."
            )
        with metric_columns[3]:
            metric_card("High performance", "Unavailable", "G3 is not present in this dataset.")
    st.caption(
        f"Active dataset: {bundle.metadata.file_name}. Performance bands use the current "
        "sidebar thresholds; the high-performer share uses records with classifiable G3 values."
    )

st.markdown(
    '<div class="el-home-section-heading"><div><span class="el-home-eyebrow">'
    "THE PROCESS</span><h2>How EduLens Works</h2></div>"
    '<span class="el-home-section-note">A clear path from observations to insight</span></div>',
    unsafe_allow_html=True,
)

pipeline = (
    ("▤", "DATA"),
    ("∑", "STATISTICS"),
    ("◉", "PROBABILITY"),
    ("⌁", "MODELLING"),
    ("✳", "INSIGHTS"),
)
pipeline_columns = st.columns(len(pipeline))
for column, (icon, label) in zip(pipeline_columns, pipeline, strict=True):
    with column:
        st.markdown(
            f'<div class="el-home-pipeline-step"><span aria-hidden="true">{icon}</span>'
            f"<strong>{label}</strong></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="el-home-section-heading"><div><span class="el-home-eyebrow">'
    "EXPLORE THE TOOLKIT</span><h2>One dataset. Multiple perspectives.</h2></div>"
    "</div>",
    unsafe_allow_html=True,
)

features = (
    (
        "📊",
        "Descriptive Statistics",
        "Summarize selected variables with distribution-aware statistics and clear context.",
        pages["descriptive"],
    ),
    (
        "🎲",
        "Probability and Bayes",
        "Calculate empirical event probabilities and follow Bayes’ theorem through its components.",
        pages["probability"],
    ),
    (
        "🧪",
        "Hypothesis Testing",
        "Compare groups with assumption checks, effect sizes, and neutral evidence language.",
        pages["hypothesis"],
    ),
    (
        "📈",
        "Regression",
        "Estimate conditional associations and inspect model diagnostics.",
        pages["regression"],
    ),
    (
        "🎯",
        "Probability Prediction",
        "Explore estimated Low, Medium, and High performance probabilities with model caveats.",
        pages["prediction"],
    ),
    (
        "🗂️",
        "CSV Upload",
        "Inspect a validated CSV in memory alongside the supplied course datasets.",
        pages["explorer"],
    ),
)

for row_start in (0, 3):
    feature_columns = st.columns(3)
    for column, (icon, title, description, page) in zip(
        feature_columns,
        features[row_start : row_start + 3],
        strict=True,
    ):
        with column, st.container(border=True):
            st.markdown(
                f'<div class="el-home-feature-icon" aria-hidden="true">{icon}</div>'
                f'<h3 class="el-home-feature-title">{title}</h3>'
                f'<p class="el-home-feature-copy">{description}</p>',
                unsafe_allow_html=True,
            )
            st.page_link(page, label=f"Open {title}", icon="🔎")

st.markdown(
    """
    <section class="el-home-honesty">
      <div>
        <span class="el-home-eyebrow">STATISTICAL HONESTY</span>
        <h2>Evidence with context.</h2>
      </div>
      <ul>
        <li><span aria-hidden="true">◇</span> Association is not causation.</li>
        <li><span aria-hidden="true">◇</span> Predictions are estimates, not guarantees.</li>
        <li><span aria-hidden="true">◇</span> Model A excludes G1 and G2.</li>
      </ul>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <footer class="el-home-footer">
      <div><strong>EDULENS-TEAM</strong><span>Probability &amp; Statistics project</span></div>
      <div><span>Dataset: Cortez (2008), <em>Student Performance</em></span>
        <a href="https://doi.org/10.24432/C5TG7T">DOI 10.24432/C5TG7T</a></div>
      <div><span>Project code license</span><strong>MIT</strong></div>
    </footer>
    """,
    unsafe_allow_html=True,
)
