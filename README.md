<div align="center">

# 🔍 EduLens
### Student Academic Performance Intelligence

*Discover the factors. Understand the patterns. Predict the probability.*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-In%20development-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

> **Status:** Phase 3 (descriptive statistics). Exploratory analysis and UI pages have not been
> implemented; sections below describe the approved plan. No analysis results are reported.

## About

EduLens is an interactive statistical analysis platform that explores **which factors are statistically associated with student academic performance**, and uses probability and statistical models to estimate performance outcomes. It was built as a Probability & Statistics course project.

Learning path: **Data → Descriptive Statistics → Probability → Hypothesis Testing → Correlation → Regression → Probability Prediction.**

EduLens separates *correlation*, *statistical association*, *prediction* and *causation*. It never claims that a factor causes a grade.

## Screenshots

| Overview | Probability | Hypothesis Testing | Prediction |
|---|---|---|---|
| _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |

_Screenshots will be added after the UI is built._

## Features

- Dataset explorer with cleaning report (missing values, duplicates, outliers, before/after)
- Descriptive statistics for user-selected variables
- Exploratory visualizations (histograms, KDE, box, violin, scatter, heatmap)
- Pearson and Spearman correlation with a selectable target
- Probability calculator: P(A), P(B), P(A|B) with a configurable performance threshold
- Bayes' theorem walkthrough using dataset frequencies
- Hypothesis tests: t-test, one-way ANOVA (+ post-hoc), chi-square, each with assumptions
- Confidence intervals (default 95%, adjustable)
- Multiple linear regression with diagnostics
- Estimated probabilities of Low / Medium / High performance
- Model A (no G1/G2) vs Model B (with G1/G2) comparison
- CSV upload for other datasets (e.g., a future college survey)

## Statistical Methods

Mean, median, mode, variance, SD, quartiles, IQR, skewness · Pearson & Spearman correlation · conditional probability · Bayes' theorem · normal/binomial/empirical distributions · confidence intervals · Welch's t-test · ANOVA · chi-square · OLS regression · multinomial logistic regression · accuracy, precision, recall, F1.
Details: [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

## Architecture

Self-contained Streamlit app; analysis logic in `src/`, UI in `views/`. See [ARCHITECTURE.md](ARCHITECTURE.md) for Mermaid diagrams.

## Tech Stack

Python · Pandas · NumPy · SciPy · Statsmodels · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · OpenPyXL · Git/GitHub

## Dataset

**Student Performance** (Cortez, 2008), UCI Machine Learning Repository, CC BY 4.0.
https://archive.ics.uci.edu/dataset/320/student+performance · DOI: 10.24432/C5TG7T
See [DATASET.md](DATASET.md).

## Installation

> Commands below are the planned workflow and will be validated in Phase 11.

```bash
git clone <REPOSITORY_URL>        # placeholder: repository URL not yet created
cd edulens
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For a fresh setup, place the UCI course CSV files in `data/raw/`; see
[data/README.md](data/README.md) for the official download source and file instructions.

## Running Locally

```bash
streamlit run app.py
```

## Usage

1. Choose a dataset in the sidebar (UCI or upload a CSV).
2. Walk through the pages from **Overview** to **Model Evaluation**.
3. Adjust the performance threshold, confidence level and α in **Settings**.
4. Read the interpretation and assumptions shown with each result.

## Project Structure

```
edulens/
├── app.py
├── views/        # page UI
├── src/          # statistics, probability, models, visualization, UI helpers
├── data/         # raw / processed
├── models/  assets/  tests/  docs/
└── *.md          # PRD, ARCHITECTURE, DATASET, STATISTICAL_METHODS, DESIGN, REQUIREMENTS
```

## Example Analysis

_To be written after implementation, using actual outputs from the app._

## Results

_Placeholder. No metrics are reported until the analysis has been run and verified._

## Limitations

- Two Portuguese secondary schools; findings may not generalize.
- Observational data: associations, not causes.
- Self-reported variables; ordinal scales.
- Predictions are probabilistic estimates and unsuitable for high-stakes decisions.

## Ethical Use

Performance is multifactorial; predictions are estimates; do not label students or use EduLens for grading, admissions or similar decisions. Survey data collection (future) must be anonymous, minimal and consent-based.

## Future Improvements

College survey dataset · Math + Portuguese combined analysis (after checking overlap) · Bayesian models · calibration plots · PDF report export.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Contributors

_Placeholder: team member names._

## License

Project code license: MIT. UCI data is CC BY 4.0; give attribution as described in
[DATASET.md](DATASET.md).
