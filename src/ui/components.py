"""Accessible, escaped UI primitives shared by EduLens pages."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from html import escape
from typing import Literal

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.assumptions import CheckResult

StatusKind = Literal["info", "notice", "evidence", "no-evidence"]
DeltaKind = Literal["positive", "negative", "neutral"]


@contextmanager
def glass_card(*, label: str | None = None) -> Iterator[DeltaGenerator]:
    """Provide a bordered glass-surface container for composing Streamlit content."""
    with st.container(border=True) as container:
        if label:
            st.markdown(
                f'<span class="el-sr-only">{escape(label)}</span>',
                unsafe_allow_html=True,
            )
        yield container


def metric_card(
    label: str,
    value: object,
    note: str = "",
    *,
    delta: str | None = None,
    delta_kind: DeltaKind = "neutral",
) -> None:
    """Render a computed metric with optional explicit text and arrow for a comparison."""
    delta_html = ""
    if delta is not None:
        if delta_kind not in {"positive", "negative", "neutral"}:
            raise ValueError("delta_kind must be 'positive', 'negative', or 'neutral'.")
        arrow = {"positive": "↑", "negative": "↓", "neutral": "↔"}[delta_kind]
        delta_html = (
            f'<p class="el-delta el-delta-{delta_kind}">'
            f'<span aria-hidden="true">{arrow}</span><span>{escape(delta)}</span></p>'
        )
    st.markdown(
        (
            '<section class="el-card el-metric" aria-label="'
            + escape(label, quote=True)
            + '"><p class="el-label">'
            + escape(label)
            + '</p><p class="el-value">'
            + escape(str(value))
            + "</p>"
            + delta_html
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
    if kind not in {"info", "notice", "evidence", "no-evidence"}:
        raise ValueError("Unsupported status kind.")
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


def pill_tabs(labels: Sequence[str]) -> list[DeltaGenerator]:
    """Render Streamlit tabs styled as a labelled pill segment control."""
    if not labels:
        raise ValueError("Pill tabs require at least one label.")
    return st.tabs(list(labels))


def pill_range_selector(
    label: str,
    options: Sequence[str],
    *,
    key: str,
    default: str | None = None,
) -> str:
    """Render a labelled horizontal pill selector and return the selected option."""
    if not options:
        raise ValueError("A pill selector requires at least one option.")
    if default is not None and default not in options:
        raise ValueError("The default selection must be one of the supplied options.")
    return st.radio(
        label,
        options=options,
        index=options.index(default) if default is not None else 0,
        key=key,
        horizontal=True,
    )


def assumptions_panel(
    checks: Sequence[CheckResult],
    *,
    alternatives: Sequence[str] = (),
    title: str = "Assumptions and checks",
) -> None:
    """Display check status, explanatory text, and alternatives accessibly."""
    with st.expander(title):
        if not checks:
            st.info("No assumption checks are available for this result.")
        for check in checks:
            icon = "✔" if check.status == "ok" else "⚠"
            st.markdown(
                (
                    '<p class="el-assumption-line"><strong>'
                    f'<span aria-hidden="true">{icon}</span> {escape(check.name)} — '
                    f"{escape(check.status.title())}</strong><br>"
                    f"{escape(check.message)}</p>"
                ),
                unsafe_allow_html=True,
            )
        if alternatives:
            st.markdown("**Suggested alternatives or follow-up**")
            items = "".join(f"<li>{escape(alternative)}</li>" for alternative in alternatives)
            st.markdown(
                f'<ul class="el-assumption-alternatives">{items}</ul>', unsafe_allow_html=True
            )


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
