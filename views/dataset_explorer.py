"""Dataset inspection, filtering, quality reporting, and cleaned-data export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import DatasetBundle


def _visible_frame(frame: pd.DataFrame, report_missing: bool = False) -> pd.DataFrame:
    """Apply UI search and one-column category filtering without modifying source data."""
    search = st.text_input(
        "Search values",
        placeholder="Type text to match rows in the selected columns",
        key="el_explorer_search",
    ).strip()
    search_columns = st.multiselect(
        "Search columns",
        options=list(frame.columns),
        default=list(frame.columns[: min(5, len(frame.columns))]),
        key="el_explorer_search_columns",
    )
    filter_column = st.selectbox(
        "Filter by column",
        options=["No filter", *frame.columns.tolist()],
        key="el_explorer_filter_column",
    )
    filtered = frame
    if search and search_columns:
        text = (
            filtered[search_columns]
            .astype("string")
            .fillna("")
            .apply(lambda column: column.str.contains(search, case=False, regex=False))
        )
        filtered = filtered.loc[text.any(axis=1)]
    if filter_column != "No filter":
        normalized = filtered[filter_column].astype("string").fillna("(Missing)")
        choices = sorted(normalized.unique().tolist())
        selected = st.multiselect(
            "Keep values",
            options=choices,
            default=choices,
            key="el_explorer_filter_values",
        )
        filtered = filtered.loc[normalized.isin(selected)]
    if report_missing and frame.isna().any().any():
        st.caption("Search and filters apply to the raw uploaded/source values.")
    return filtered


selected = st.session_state.get("edulens_dataset_bundle")
bundle = selected if isinstance(selected, DatasetBundle) else None
st.title("Dataset Explorer")
st.markdown(
    '<p class="el-caption">Inspect the selected course or uploaded CSV. Filtering is for '
    "display only; the source dataset remains unchanged.</p>",
    unsafe_allow_html=True,
)

if bundle is None:
    st.info("Select or upload a dataset from the sidebar to explore its contents.")
else:
    st.caption(
        f"{bundle.metadata.file_name} · {bundle.metadata.row_count:,} rows × "
        f"{bundle.metadata.column_count:,} columns · "
        f"encoding {bundle.metadata.encoding} · delimiter {bundle.metadata.delimiter!r}"
    )
    preview_tab, schema_tab, missing_tab, cleaning_tab = st.tabs(
        ["Preview / search / filter", "Columns and dtypes", "Missing values", "Cleaning report"]
    )
    with preview_tab:
        visible = _visible_frame(bundle.raw)
        st.caption(f"Showing {len(visible):,} of {len(bundle.raw):,} source rows.")
        st.dataframe(visible, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cleaned CSV",
            data=bundle.clean.to_csv(index=False).encode("utf-8"),
            file_name=f"{bundle.metadata.file_name.rsplit('.', 1)[0]}-cleaned.csv",
            mime="text/csv",
            help="Downloads the non-destructively cleaned table, including the G3-zero flag "
            "when G3 is available.",
        )
    with schema_tab:
        schema = pd.DataFrame(
            {
                "Column": list(bundle.clean.columns),
                "Data type": [str(dtype) for dtype in bundle.clean.dtypes],
            }
        )
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.caption(
            f"{len(bundle.clean.columns):,} columns after cleaning. "
            "G3_zero_flag is a derived nullable indicator where G3 exists."
        )
    with missing_tab:
        missing = pd.DataFrame(
            {
                "Column": list(bundle.cleaning_report.missing_by_column),
                "Missing values": list(bundle.cleaning_report.missing_by_column.values()),
            }
        )
        st.metric("Missing source cells", f"{bundle.cleaning_report.missing_cells:,}")
        if missing.empty:
            st.success("No missing values were found in the source dataset.")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)
        st.caption("Missingness is reported from the source table before any derived column.")
    with cleaning_tab:
        report = bundle.cleaning_report
        before, after = st.columns(2)
        with before:
            st.subheader("Before")
            st.metric("Rows", f"{report.rows_before:,}")
            st.metric("Columns", f"{report.columns_before:,}")
            st.metric("Missing cells", f"{report.missing_cells:,}")
        with after:
            st.subheader("After")
            st.metric("Rows", f"{report.rows_after:,}")
            st.metric("Columns", f"{report.columns_after:,}")
            st.metric(
                "Exact duplicate rows (reported, not removed)", f"{report.exact_duplicate_rows:,}"
            )
        st.caption(
            "Cleaning is non-destructive: source rows, duplicates, missing values, and outliers "
            "are retained. A G3-zero flag is added when possible."
        )
        if report.numeric_summary_before:
            st.subheader("Numeric summaries before / after")
            summary_rows = []
            for column in report.numeric_summary_before:
                summary_rows.append(
                    {
                        "Column": column,
                        "Before count": report.numeric_summary_before[column]["count"],
                        "Before mean": report.numeric_summary_before[column]["mean"],
                        "After count": report.numeric_summary_after.get(column, {}).get("count"),
                        "After mean": report.numeric_summary_after.get(column, {}).get("mean"),
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
