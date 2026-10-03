"""Streamlit AppTest smoke coverage for every routed page and invalid uploads."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE_PATHS = (
    "views/overview.py",
    "views/dataset_explorer.py",
    "views/descriptive_statistics.py",
    "views/exploratory_analysis.py",
    "views/correlation.py",
    "views/probability.py",
    "views/hypothesis_testing.py",
    "views/regression.py",
    "views/prediction.py",
    "views/model_evaluation.py",
    "views/about.py",
)
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _messages(elements: Iterable[object]) -> list[str]:
    """Collect visible Streamlit message strings from AppTest elements."""
    return [str(element.value) for element in elements]


@pytest.mark.parametrize("page_path", PAGE_PATHS)
def test_every_page_loads_with_portuguese_dataset(page_path: str) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=300).run()

    app.switch_page(page_path).run(timeout=300)

    assert not app.exception, [element.message for element in app.exception]
    assert app.title or app.header or app.markdown
    assert app.session_state["edulens_dataset_bundle"].metadata.file_name == "student-por.csv"


@pytest.mark.parametrize("page_path", PAGE_PATHS)
def test_every_page_shows_friendly_error_for_invalid_csv(page_path: str) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    app.selectbox(key="el_dataset_choice").set_value("Upload CSV").run(timeout=120)
    app.file_uploader[0].set_value(("invalid.csv", b"value\n1\n", "text/csv")).run(timeout=120)

    assert not app.exception
    assert any("at least 20 are required" in message for message in _messages(app.error))

    app.switch_page(page_path).run(timeout=120)

    assert not app.exception, [element.message for element in app.exception]
    visible_messages = _messages(app.error) + _messages(app.warning) + _messages(app.info)
    assert any("at least 20 are required" in message for message in visible_messages)
