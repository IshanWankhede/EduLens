# Contributing to EduLens

## Branching

Create a short-lived branch for each change, using a descriptive name such as
`feat/add-data-validation` or `fix/csv-encoding-check`. Keep changes focused and do not include
unrelated work.

## Commit style

Use a short imperative subject with a conventional prefix, for example:

- `docs: clarify local setup`
- `feat: add data validation`
- `test: cover probability edge cases`

## Development setup

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Tests and code style

Run the tests from the repository root:

```bash
python -m pytest
```

Install the configured style tools if they are not already available, then run:

```bash
python -m pip install ruff black
ruff check .
black --check .
```

The shared formatting line length is 100 characters.

## Analysis integrity

Never invent dataset facts, statistical results, accuracy values, URLs, citations, or screenshots.
Displayed results must be computed from loaded data. Describe observational findings as
associations, not causes, and keep statistical logic in `src/` rather than in Streamlit views.
