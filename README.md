<div align="center">

# EduLens
### Student Academic Performance Statistics

*Explore distributions, associations, statistical tests, and estimated performance probabilities.*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

EduLens is a Probability & Statistics course project for exploring the UCI Student Performance
dataset. It reports descriptive statistics, empirical probabilities, associations, hypothesis
tests, regression estimates and classification performance. The observational data do not
establish causation, and individual model probabilities are not guaranteed outcomes.

## Features

- Overview and dataset explorer with search, filters, data types, missingness, cleaning report and
  cleaned-CSV download.
- Selectable descriptive statistics, EDA charts and Pearson/Spearman association analysis.
- Conditional-probability calculator, Bayes walkthrough and normal, binomial and empirical
  distribution summaries.
- Welch/pooled t-test, one-way ANOVA with Tukey HSD, and chi-square test with assumptions and
  alternative-test guidance.
- OLS regression with coefficient intervals, residual and influence diagnostics, and optional
  HC3 standard errors.
- Low/Medium/High probability estimates, leakage-safe Model A and explicitly grade-informed
  Model B, holdout metrics, stratified cross-validation and a majority-class baseline.
- Portuguese primary dataset, separately selectable Mathematics dataset, and in-memory CSV upload.

The fixed default grade bands are Low `< 10`, Medium `10–13` inclusive, and High `>= 14`; the
sidebar allows the band cutoffs and random seed to be changed. The model evaluation uses a
configured stratified 20% holdout and five-fold cross-validation by default. See
[DATASET.md](DATASET.md), [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md) and
[REQUIREMENTS.md](REQUIREMENTS.md) for measured dataset facts, method details, and implementation
status.

## Screenshots

Screenshots have not been captured. The table below is intentionally left for manual additions.

| Overview | Probability | Hypothesis Testing | Prediction |
|---|---|---|---|
| _capture manually_ | _capture manually_ | _capture manually_ | _capture manually_ |

See [docs/screenshots/README.md](docs/screenshots/README.md) for the exact capture list and
filenames. No screenshots are generated or represented as actual app captures by this project.

## Requirements

- Python 3.11
- The supplied local files in `data/raw/` for offline use. Their inventory, source and checksums
  are documented in [data/README.md](data/README.md).
- Internet access is not required when the local dataset files are present.

## Install and run

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. The app starts with the local
Portuguese dataset selected. Select Mathematics or upload a CSV from the sidebar to change data.
Uploaded files are validated and held in memory; EduLens does not write them to disk.

## Test and style checks

```powershell
python -m pytest --cov=src --cov-report=term-missing
ruff check .
black --check .
```

The test suite includes hand-checkable statistical fixtures, cross-checks against scientific
Python libraries, real-data assertions limited to verified properties, and Streamlit AppTest
smoke coverage for every route using both the Portuguese data and an invalid CSV.

## Project structure

```text
app.py              Streamlit setup, navigation, shared sidebar and dataset loading
views/              UI scripts for the 11 pages
views/_analysis.py  cached model analysis helpers
src/                data validation, statistical methods, models and chart builders
data/raw/           supplied UCI source data
models/             generated model artifacts (git-ignored)
tests/              unit, real-data and AppTest coverage
docs/               example analysis and manual screenshot checklist
```

The app uses `st.navigation`/`st.Page`; statistical calculations belong to `src/`. See
[ARCHITECTURE.md](ARCHITECTURE.md).

## Dataset citation

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository.
[DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). Dataset license: CC BY 4.0.
See [DATASET.md](DATASET.md) for attribution, data limitations and variable details.

## Example analysis

Reproducible outputs and the exact calls that generated them are recorded in
[docs/EXAMPLE_ANALYSIS.md](docs/EXAMPLE_ANALYSIS.md).

## Ethics and limitations

The dataset covers two Portuguese secondary schools, includes sensitive student and family
attributes, and contains observational records. Results may not generalize to other settings.
EduLens is for educational statistical exploration, not grading, admissions or other high-stakes
individual decisions. Predictions are statistical estimates, not guarantees.

## License

Project code: MIT. Dataset: CC BY 4.0; cite the source above.
