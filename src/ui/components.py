"""Accessible, escaped UI primitives shared by EduLens pages."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Literal

import streamlit as st

from src.assumptions import CheckResult

StatusKind = Literal["info", "notice", "evidence", "no-evidence"]


def metric_card(label: str, value: object, note: str = "") -> None:
    """Render a metric card with exact text and optional explanatory note."""
    st.markdown(
        (
            '<section class="el-card el-metric" aria-label="'
            + escape(label, quote=True)
            + '"><p class="el-label">'
            + escape(label)
            + '</p><p class="el-value">'
            + escape(str(value))
            + "</p>"
            + (f'<p class="el-note">{escape(note)}</p>' if note else "")
            + "</section>"
        ),
        unsafe_allow_html=True,
    )


def probability_card(label: str, percentage: float, fraction: str) -> None:
    """Render a probability with its formula/count fraction shown as text."""
    st.markdown(
        (
            '<section class="el-card el-probability" aria-label="'
            + escape(label, quote=True)
            + '"><p class="el-label">'
            + escape(label)
            + f'</p><p class="el-value">{percentage:.1f}%</p>'
            + f'<p class="el-note">{escape(fraction)}</p></section>'
        ),
        unsafe_allow_html=True,
    )


def status_chip(label: str, kind: StatusKind = "info") -> None:
    """Render semantic status with both an icon and an explicit text label."""
    icon = {
        "info": "ℹ",
        "notice": "⚠",
        "evidence": "●",
        "no-evidence": "○",
    }[kind]
    st.markdown(
        (
            f'<span class="el-status el-status-{kind}" role="status">'
            f'<span aria-hidden="true">{icon}</span> {escape(label)}</span>'
        ),
        unsafe_allow_html=True,
    )


def assumptions_panel(
    checks: Sequence[CheckResult],
    *,
    alternatives: Sequence[str] = (),
    title: str = "Assumptions and checks",
) -> None:
    """Display check status icons and readable text in an accessible collapsible panel."""
    with st.expander(title):
        if not checks:
            st.info("No assumption checks are available for this result.")
        for check in checks:
            icon = "✔" if check.status == "ok" else "⚠"
            st.markdown(
                f"**{icon} {escape(check.name)} — {escape(check.status.title())}**  \n"
                f"{escape(check.message)}"
            )
        if alternatives:
            st.markdown("**Suggested alternatives or follow-up**")
            for alternative in alternatives:
                st.markdown(f"- {escape(alternative)}")


def pipeline_strip(
    steps: Sequence[str] = ("DATA", "STATISTICS", "PROBABILITY", "MODELLING", "INSIGHTS"),
) -> None:
    """Render the project pipeline as labeled, responsive steps."""
    if not steps:
        return
    items = "".join(
        f'<div class="el-pipeline-step"><span class="el-pipeline-index">{index:02d}</span>'
        f"<span>{escape(step)}</span></div>"
        for index, step in enumerate(steps, start=1)
    )
    st.markdown(
        f'<div class="el-pipeline" aria-label="Project workflow">{items}</div>',
        unsafe_allow_html=True,
    )
