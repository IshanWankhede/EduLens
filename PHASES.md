# EduLens: Phase Prompts for the Coding Agent

**How to use:**
1. Put all docs in the repo root (PRD, REQUIREMENTS, ARCHITECTURE, DATASET, STATISTICAL_METHODS, DESIGN, README, ROADMAP, requirements.txt, .gitignore).
2. Fill in the **Project Decisions** block below.
3. Paste the **Global Rules** once at the start of the agent session (or save as the agent's rules/system file).
4. Give the agent **one phase prompt at a time**. Review the output, run the tests, then move on.

---

## Project Decisions (fill in before Phase 2)

```
PERFORMANCE_BANDS: <fixed | quantile>   # if fixed: LOW < ___, MEDIUM ___–___, HIGH >= ___ (on the 0–20 scale, with justification)
PRIMARY_FILE:      <student-por.csv | student-mat.csv | both analyzed separately>
PYTHON_VERSION:    <e.g. 3.11>
LICENSE:           <e.g. MIT>
TEAM_NAMES:        <for README contributors>
```

If any value is still blank when a phase needs it, the agent must **ask** instead of guessing.

---

## Global Rules (paste once, applies to every phase)

```
You are working on EduLens, a Streamlit Probability & Statistics course project.
The docs in the repo root are the approved specification. Read them before every phase:
PRD.md, REQUIREMENTS.md, ARCHITECTURE.md, DATASET.md, STATISTICAL_METHODS.md, DESIGN.md, ROADMAP.md.

Rules:
1. Follow ARCHITECTURE.md exactly (folders: views/, src/, src/ui/, tests/, etc.). Statistical logic lives in src/, never in views/. src/ modules must not import streamlit (except documented caching wrappers).
2. Never invent statistical results, accuracy numbers, dataset values, URLs, citations or screenshots. Every number shown must be computed from data.
3. Never claim causation. Use "association", "model contribution", "estimated probability".
4. G1/G2 must never enter Model A. G3 is never a feature. Preprocessing is inside sklearn Pipelines (no leakage).
5. Every statistical function: typed signature, docstring (purpose, assumptions), returns a dataclass/dict, handles edge cases (empty data, tiny groups, P(B)=0, NaNs) with friendly custom exceptions.
6. Write pytest tests with small hand-checkable fixtures and cross-check against SciPy/statsmodels. Run the tests and show results. Do not claim tests pass unless you ran them.
7. Keep code modular, typed, readable. Use the fixed random seed from src/config.py.
8. If DATASET.md has [VERIFY] items relevant to your phase, verify them from the real files and update the doc. If a project decision is missing, ask me.
9. Do not do work from later phases. Do not refactor unrelated files.
10. At the end of each phase: (a) list files created/changed, (b) show how to run/test, (c) list open questions or deviations from the docs, (d) STOP and wait for my approval.
```

---

## Phase 1: Documentation + Architecture (already done)

Docs are generated. **Agent prompt (one-time repo setup):**

```
Phase 1 – Repository setup.
Read all docs in the repo root. Then:
1. Create the folder structure from ARCHITECTURE.md §3 with empty __init__.py files where needed, and .gitkeep files in data/raw, data/processed, models, assets/logo, assets/images.
2. Create docs/ and move nothing out of the root. Add CONTRIBUTING.md (branching, commit style, how to run tests, code style: ruff + black, "no invented results" rule).
3. Create pyproject.toml or ruff/black config (line length 100) and pytest config (testpaths = tests).
4. Create src/config.py with: RANDOM_SEED, DEFAULT_ALPHA=0.05, DEFAULT_CONFIDENCE=0.95, LEAKY_COLUMNS=["G1","G2","G3"], column role lists (numeric, ordinal, nominal, binary) taken ONLY from DATASET.md §3, and placeholders for performance bands read from the Project Decisions in PHASES.md.
5. Create a minimal app.py that runs and shows the EduLens title (no real pages yet).
6. Set up a virtual environment instructions check: pip install -r requirements.txt and streamlit run app.py must work.
Stop for approval after listing what you did.
```

---

## Phase 2: Dataset Ingestion + Cleaning

```
Phase 2 – Dataset ingestion and cleaning.
Implement src/data_loader.py, src/validation.py, src/preprocessing.py and tests.

Requirements:
1. Download/read the official UCI Student Performance files (link in DATASET.md) into data/raw/. Verify and report: files inside the zip, delimiter, encoding, row/column counts per file, G3 range, count of G3 == 0, absences range, any missing values, exact duplicate rows, and whether students overlap between the math and Portuguese files (and how identifiable). Update DATASET.md, replacing every [VERIFY] with verified facts, and write data/README.md including file checksums.
2. data_loader.py: load UCI file(s) and user-uploaded CSV; return a DatasetBundle dataclass (raw, clean, metadata, column roles, cleaning report). Cache-ready (decorators applied in a thin wrapper; the pure functions stay Streamlit-free).
3. validation.py: friendly errors for invalid CSV, wrong delimiter/encoding, empty file, too few rows, missing required columns (G3 for the UCI schema), wrong dtypes. Custom exceptions: DataValidationError, InsufficientDataError.
4. preprocessing.py: missing-value report, duplicate report, dtype validation, outlier inspection via the IQR rule (report only, do not delete), categorical encoding helpers (binary → 0/1, nominal → dummies with reference level documented), numeric validation, before/after cleaning statistics, and a function that derives the Low/Medium/High performance category from G3 using the configured thresholds (fixed or quantile per Project Decisions; document the method in the returned object).
5. Tests: small fixture DataFrames for each check, plus a test that loads the real file and asserts only facts you verified.
Do not build any UI pages yet.
```

---

## Phase 3: Descriptive Statistics

```
Phase 3 – Descriptive statistics.
Implement src/descriptive_stats.py and tests/test_descriptive_stats.py.

1. summary_table(df, columns): mean, median, mode (all ties), min, max, range, sample variance (ddof=1), SD, Q1, Q2, Q3, IQR, skewness, count, missing count.
2. group_summary(df, value_col, group_col): same stats per group, with group sizes and a CI for each group mean at a configurable confidence level.
3. frequency_table(df, col): counts and percentages for categorical/ordinal columns.
4. Document in docstrings which quartile method and skewness formula pandas uses.
5. Tests with hand-computed values from a tiny fixture and cross-checks vs NumPy/SciPy.
6. Edge cases: constant column, single row, all-NaN, non-numeric column → friendly exceptions.
No UI yet.
```

---

## Phase 4: EDA + Visualization

```
Phase 4 – Exploratory analysis and visualization.
Implement src/visualization.py (figure builders returning Plotly or Matplotlib figures, no Streamlit calls) and tests that figures build without error.

Charts: histogram, KDE/distribution, box, violin, bar/count, scatter (with optional trend line and clearly labelled), correlation heatmap (labelled colorbar), grouped comparison chart.
Preset relationship builders: studytime vs G3, absences vs G3, failures vs G3, G1 vs G3, G2 vs G3, famsup vs G3, Medu/Fedu vs G3, health vs G3, freetime vs G3.

Rules (see DESIGN.md §7): shared theme function, colorblind-safe palette, every chart has a title, axis labels, optional caption text; bar charts start at zero; probabilities on a 0–100% axis; don't rely on color alone (use markers/patterns). Use Plotly for interactivity; Matplotlib/Seaborn for static diagnostics. Include a "how to read this" caption string for each builder.
Do not create Streamlit pages yet.
```

---

## Phase 5: Probability Analysis

```
Phase 5 – Probability analysis.
Implement src/probability.py, src/correlation.py and their tests.

probability.py:
1. conditional_probability(df, event_a, event_b) returning counts n(A), n(B), n(A∩B), n, P(A), P(B), P(A∩B), P(A|B), and a Wilson CI for P(A|B). Events are defined by simple condition objects (column, operator/category, threshold). Handle P(B)=0 and very small n(B) with clear messages.
2. independence_check: compare P(A|B) with P(A); report difference and note it is descriptive, not causal.
3. bayes(df, event_a, event_b): prior P(B), likelihood P(A|B), evidence P(A) via the law of total probability, posterior P(B|A); return all intermediate values and verify equality with the direct frequency n(A∩B)/n(A) in a test.
4. distributions: fit normal (μ̂, σ̂), binomial for a binary event (p̂, n), empirical CDF; goodness-of-fit using Q–Q data, Shapiro–Wilk, KS (note estimated parameters make KS p-values approximate); return an honest verdict ("reasonable" / "poor fit") with the evidence. Never force a fit.
correlation.py:
5. Pearson and Spearman matrices, p-values, Fisher-z CI for Pearson, ranked association table for a selectable target, strength labels using documented thresholds; automatically recommend Spearman for ordinal columns per DATASET.md.
Tests: tiny hand-calculable fixtures for conditional probability and Bayes, SciPy cross-checks for correlation.
```

---

## Phase 6: Hypothesis Testing

```
Phase 6 – Hypothesis testing.
Implement src/assumptions.py, src/hypothesis_tests.py and tests.

assumptions.py: Shapiro–Wilk (per group/residuals), Levene, expected-count check for chi-square, VIF helper, minimum group-size check. Each returns a result with pass/warn status AND plain-English text.

hypothesis_tests.py (follow STATISTICAL_METHODS.md §14–17):
1. independent_t_test: Welch default, optional pooled; returns H0/H1, t, df, p, alpha, CI for mean difference (configurable confidence), Cohen's d (Hedges' correction), decision text using "Sufficient / Insufficient evidence against H0" wording, interpretation text, assumptions list with check results, and a suggested alternative (Mann–Whitney U) if assumptions fail. Also provide Mann–Whitney as a callable alternative.
2. one_way_anova: F, df, p, group means with CIs, eta-squared, Levene/Shapiro checks, Welch ANOVA and Kruskal–Wallis alternatives, Tukey HSD post-hoc when significant (document any alternative you choose and why).
3. chi_square_independence: observed table, expected table, standardized residuals, chi2, df, p, Cramér's V, expected-count warning, Fisher's exact for 2x2 when needed; state the Yates-correction setting.
4. Group-creation helper for splitting numeric/ordinal variables (e.g., studytime >= 3 vs < 3); the cutoff must be returned in the result so the UI can display it.
5. A result dataclass with fields matching the UI flow: TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION.
Never use "accept H0", "prove", or causal language. Tests: cross-check with SciPy/statsmodels on fixtures; edge cases (tiny groups, zero variance).
```

---

## Phase 7: Regression

```
Phase 7 – Multiple linear regression.
Implement src/regression.py and tests (follow STATISTICAL_METHODS.md §18).

1. fit_ols(df, target, predictors, exclude_leaky=True): statsmodels OLS with a documented encoding scheme (binary 0/1, nominal dummies with reference level, ordinal treated as numeric or categorical by parameter). If target is G3 and exclude_leaky=True, G1/G2 are rejected with a clear message; Model B regression is allowed only with an explicit flag and a label in the result.
2. Return: coefficient table (coef, SE, t, p, CI at configurable level), R², adjusted R², F-stat and p, residual SE, n, plain-English coefficient interpretations ("holding other included variables constant… associated with…").
3. Diagnostics data: fitted values, residuals, standardized residuals, leverage, Cook's distance, VIF table, Breusch–Pagan, Shapiro–Wilk on residuals; assumptions list with statuses and alternatives (HC robust SE option, transformations).
4. Provide a robust-SE option (HC3).
5. Test: compare against a direct statsmodels fit and a NumPy normal-equation solution on a small fixture; test that G1/G2 are rejected for Model A regression.
No UI yet.
```

---

## Phase 8: Prediction

```
Phase 8 – Classification and probability prediction.
Implement src/prediction.py and tests (follow ARCHITECTURE.md §8 and STATISTICAL_METHODS.md §19–20).

1. Feature sets: MODEL_A_FEATURES (no G1/G2/G3) and MODEL_B_FEATURES (A + G1 + G2) defined from config only.
2. build_pipeline(feature_set): sklearn Pipeline (ColumnTransformer: impute → one-hot/scale → multinomial LogisticRegression). Report regularization settings.
3. Performance categories from G3 using the configured thresholds. If quantile thresholds are used, compute them from the training split only (or document clearly).
4. Stratified train/test split with fixed seed; stratified k-fold CV (mean ± SD).
5. Metrics: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, majority-class baseline, optional log-loss/Brier.
6. predict_proba_for_profile(profile_dict): returns P(Low), P(Medium), P(High) summing to 1.
7. compare_models(): Model A vs Model B using identical splits/folds; return the differences with spread. Do not assert in code or text that Model B is better; report what the data shows.
8. Coefficient/odds-ratio view with the caveat "model contribution, not causation"; optional statsmodels MNLogit for p-values if it converges (document if it doesn't).
9. Tests: Model A feature names (after encoding) contain no G1/G2/G3; probabilities sum to 1; deterministic with the seed; no leakage (preprocessing fit only on training data).
10. Save fitted pipelines to models/ only through an explicit function (models/ is git-ignored).
```

---

## Phase 9: Streamlit UI

```
Phase 9 – Streamlit UI.
Build app.py, views/*, src/ui/theme.py, src/ui/components.py, assets/styles/edulens.css, .streamlit/config.toml, following DESIGN.md and ARCHITECTURE.md §5, §11 exactly.

1. app.py: set_page_config, load theme, custom sidebar (logo/name, dataset selector UCI/upload, CSV upload, navigation for the 11 pages, settings expander: confidence level, alpha, performance thresholds, seed; dataset citation footer), routing via st.navigation/st.Page or a router. Use views/ (not pages/).
2. Pages: Overview (hero "Understand What Shapes Student Performance", animated metric cards, How EduLens Works strip), Dataset Explorer (tabs: preview/search/filter, columns & dtypes, missing values, cleaning before/after, download cleaned CSV), Descriptive Statistics, Exploratory Analysis, Correlation, Probability (calculator + Bayes walkthrough + distributions), Hypothesis Testing (test selector + TEST→STATISTIC→P-VALUE→DECISION→INTERPRETATION flow + assumptions panel), Regression, Prediction (inputs, probability visualization, verbatim disclaimer), Model Evaluation (confusion matrix, metrics, CV, Model A vs B), About (methods, ethics, limitations, dataset citation).
3. Pages call src/ functions only; no statistics in views.
4. Caching: st.cache_data for loading/cleaning/deterministic stats, st.cache_resource for fitted models; invalidate when the dataset changes.
5. Error handling: catch custom exceptions and show friendly st.error/st.warning; never show raw tracebacks; log details.
6. CSS: variables from DESIGN.md §2, fade-in, hover, progress transitions, respect prefers-reduced-motion; escape user-provided text in HTML; neutral decision wording (no pass/fail), icon + text for status.
7. Accessibility and responsiveness per DESIGN.md §9–10 (labels, chart titles, no color-only meaning, stacking on narrow screens).
8. The Prediction page must always display: "This is a statistical estimate based on the selected dataset and model. It should not be interpreted as a guaranteed prediction of an individual's academic outcome."
Do not hard-code any statistic or result in the UI.
```

---

## Phase 10: Testing + Documentation

```
Phase 10 – Testing and documentation.
1. Run the full pytest suite; fix failures; report coverage by module (do not claim a number you did not measure).
2. Add Streamlit AppTest smoke tests for every page with the UCI dataset and with an invalid CSV (expect friendly errors).
3. Audit: grep the repo for "cause", "causes", "proves", "guarantee" in UI strings and fix misuse; confirm G1/G2 absent from Model A; confirm no hard-coded results.
4. Run ruff/black; fix issues.
5. Update docs to match actual behavior: README (real run instructions, feature list), STATISTICAL_METHODS.md (any deviations), DATASET.md, ARCHITECTURE.md (if structure changed), REQUIREMENTS.md (mark each FR/NFR/SR/PR/UI as Done/Partial/Not done with a note).
6. Do NOT fabricate screenshots. Create a docs/screenshots/ folder and list exactly which screenshots I should capture manually and where to put them; leave README placeholders until I add them.
7. Produce docs/EXAMPLE_ANALYSIS.md by running the app's own functions on the UCI data and recording the real outputs (with the code that produced them). Label the parameters (thresholds, seed, alpha).
```

---

## Phase 11: Deployment Preparation

```
Phase 11 – Deployment preparation.
1. In a fresh virtual environment, install from requirements.txt, run tests, run the app. Report any dependency conflicts and fix them.
2. Generate requirements.lock.txt with exact versions from the working environment; keep requirements.txt with lower bounds.
3. Ensure the app starts without internet if data is already in data/raw (document the download step otherwise); make sure fonts have fallbacks.
4. Add a short docs/DEPLOYMENT.md: local run, optional Streamlit Community Cloud steps (as instructions only; do not invent a deployment URL), required files, secrets (none expected).
5. Review .gitignore, remove stray files, confirm no secrets or personal data in the repo. Confirm the uploaded-CSV path does not persist user data.
6. Final README pass: replace placeholders that can now be filled from facts (license, team names from Project Decisions, run commands) and leave placeholders for the repo URL, screenshots and live URL.
7. Produce a final checklist against PRD.md §12 acceptance criteria, marking each item pass/fail with evidence (file/test names).
```

---

## Between-Phase Review Checklist (for you)

- Do tests pass when you run them yourself?
- Any number in the output that the code did not compute?
- Any "causes" wording?
- G1/G2 absent from Model A?
- Did the agent stay inside the phase?
- Did it update docs where it deviated?
