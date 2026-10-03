# EduLens: Product Requirements Document

**Name:** EduLens, Student Academic Performance Intelligence
**Tagline:** *Discover the factors. Understand the patterns. Predict the probability.*
**Status:** Phase 1 (documentation), awaiting approval before implementation
**Course context:** Probability & Statistics, BTech CSE (2nd year)

---

## 1. Product Overview

EduLens is a self-contained Streamlit application that studies one question:

> **What factors are statistically associated with student academic performance?**

It is a *statistics-first* platform. The learning path is:

`Data → Descriptive Statistics → Probability → Hypothesis Testing → Correlation → Regression → Probability Prediction`

Machine learning is used only where it supports a statistical idea (for example, multinomial logistic regression for class probabilities). Every conclusion shown in the UI must be traceable to a calculation on the loaded data.

**Primary dataset:** UCI Student Performance dataset (official page: https://archive.ics.uci.edu/dataset/320/student+performance). The 649-student file is the Portuguese-course file. Exact file names, counts and license text must be verified from the UCI page and recorded in `DATASET.md`; nothing is copied from memory.

**Secondary sources:** user-uploaded CSV, and a future college survey dataset.

## 2. Problem Statement

Students, instructors and course evaluators often talk about "what makes students do well" using anecdotes. Few tools let a learner explore this with proper probability and inference, while keeping correlation, association, prediction and causation clearly separate. EduLens fills that gap as both an analysis tool and a teaching demonstration.

## 3. Target Users

| User | Need |
|---|---|
| Course student (primary, the project team) | Demonstrate P&S concepts on real data |
| Course instructor / evaluator | Verify rigor, assumptions and traceability |
| Peer learners | Interactively explore concepts (conditional probability, Bayes, tests) |
| Future: college survey analyst | Upload a CSV and get the same analyses |

## 4. Goals

1. Show distribution, spread and shape of academic performance.
2. Quantify associations (Pearson, Spearman) with a selectable target.
3. Provide an interactive conditional probability and Bayes calculator on real frequencies.
4. Run t-test, one-way ANOVA (with post-hoc) and chi-square, each with assumptions and plain-English interpretation.
5. Fit multiple linear regression with full inference output and diagnostics.
6. Estimate P(Low), P(Medium), P(High) via a classification model.
7. Compare **Model A (no G1/G2)** against **Model B (with G1/G2)** transparently.
8. Support CSV upload with validation.
9. Deliver a polished, accessible dashboard.

## 5. Non-Goals

- Not a high-stakes decision tool (no admissions, grading, intervention targeting).
- No causal inference claims.
- No backend services, databases, React, FastAPI or Node.js in v1.
- No survey collection built into the app in v1 (only CSV ingestion).
- No hyperparameter-heavy ML model zoo.

## 6. User Stories

- As a student, I want to pick a performance threshold and see P(High) so I understand probability on real data.
- As a student, I want to choose a condition (e.g., high study time) and see P(A), P(B), P(A|B) and P(A∩B).
- As a student, I want a step-by-step Bayes calculation (prior, likelihood, evidence, posterior).
- As an evaluator, I want every test to show assumptions, statistic, p-value, CI, decision and interpretation.
- As a user, I want to upload my own CSV and map columns to roles (target, predictors).
- As a user, I want to enter a student profile and see estimated class probabilities with a disclaimer.
- As a user, I want to see how much G1/G2 improve prediction compared with factor-only models.
- As a user, I want helpful error messages, never raw stack traces.

## 7. Functional Features (summary; IDs in REQUIREMENTS.md)

| Page | Key features |
|---|---|
| Overview | Hero, metric cards (size, variables, mean, median, % High), "How EduLens Works" |
| Dataset Explorer | Preview, search, filter, dtypes, missing-value summary, before/after cleaning, cleaned download |
| Descriptive Statistics | Dynamic variable selection; mean, median, mode, min, max, range, variance, SD, quartiles, IQR, skewness; group comparison |
| Exploratory Analysis | Histogram, KDE, box, violin, bar/count, scatter, grouped charts |
| Correlation | Pearson and Spearman matrices, heatmap, ranked table, selectable target |
| Probability | Configurable threshold; P(A), P(B), P(A\|B); Bayes walkthrough; distribution fitting (normal, binomial, empirical) |
| Hypothesis Testing | t-test, ANOVA + post-hoc, chi-square; assumption checks |
| Regression | OLS with coefficients, SE, p, CI, R², adj. R², diagnostics |
| Prediction | Profile input, class probabilities, disclaimer |
| Model Evaluation | Confusion matrix, accuracy, precision, recall, F1, CV, Model A vs B |
| About | Methods, ethics, limitations |

## 8. Statistical Requirements (summary)

- Assumptions are displayed with every test and model; violations trigger a warning plus an alternative (Welch's t, Mann–Whitney U, Kruskal–Wallis, Fisher's exact, robust SE, etc.).
- Confidence level configurable, default 95%.
- Performance thresholds configurable; the default method is documented (for example, tertiles/quantiles of the loaded data, or fixed grade bands) and shown on screen. Defaults are **not** hard-coded silently.
- No distribution is forced onto a variable; goodness-of-fit is reported honestly, including "poor fit".
- Association is never described as causation.

## 9. Prediction Requirements (summary)

- **Model A:** factor-only; G1 and G2 are excluded by an explicit, tested guard.
- **Model B:** Model A features + G1 + G2.
- Stratified train/test split, fixed random seed, cross-validation, preprocessing inside a pipeline fitted on training data only (no leakage).
- Report accuracy, per-class precision, recall, F1, confusion matrix; for imbalance, also macro-F1.
- Language: "Estimated probability based on the statistical model."

## 10. Non-Functional Requirements (summary)

Responsive layout, caching of loads and model fits, graceful errors, WCAG-minded contrast, reproducibility, modular code, unit tests for statistical functions.

## 11. UI Requirements (summary)

Midnight-navy base with indigo/blue/violet/cyan accents, subtle glassmorphism, CSS variables, light CSS animations, charts always titled, no color-only meaning, neutral (not pass/fail) test status language. Full detail goes in `DESIGN.md`.

## 12. Acceptance Criteria

1. App launches with one command and loads UCI data (or a local copy of it) without manual steps beyond documented setup.
2. Every number on screen is computed from the loaded data; no hard-coded results.
3. G1/G2 provably absent from Model A (automated test).
4. Each hypothesis test shows H0/H1, statistic, p-value, α, CI, decision, interpretation and assumptions.
5. Probability page reproduces hand-calculable P(A), P(B), P(A|B) on a small test fixture.
6. Invalid CSV, empty data and tiny groups produce friendly messages.
7. Prediction page always shows the disclaimer.
8. README contains no invented URLs, metrics or screenshots.

## 13. Ethical Considerations

1. **Performance is multifactorial.** The dataset captures a small subset of influences.
2. **Association is not causation.** Observational data cannot establish that a factor *causes* grades.
3. **Predictions are estimates** with uncertainty, tied to one dataset and population.
4. **Avoid labeling students.** "High/Low" are analytic categories, not judgments of ability.
5. **No high-stakes use.** Not for grading, admissions, placement or disciplinary action.
6. **Sensitive attributes** (sex, age, address, family background) appear in the data. They are treated cautiously: analyzed descriptively only where justified, and any model use is documented with fairness caveats.
7. **Survey data (future):** anonymize responses; do not collect names, email addresses, phone numbers or unnecessary personal information; obtain informed consent before primary data collection; explain purpose, voluntariness and data handling; follow the institution's rules.
8. **Uploaded CSVs** are processed in-session and not persisted by the app by default.

## 14. Development Phases

| Phase | Scope |
|---|---|
| 1 | Documentation and architecture |
| 2 | Dataset ingestion and cleaning |
| 3 | Descriptive statistics |
| 4 | EDA and visualization |
| 5 | Probability analysis |
| 6 | Hypothesis testing |
| 7 | Regression |
| 8 | Prediction |
| 9 | Streamlit UI |
| 10 | Testing and documentation |
| 11 | Deployment preparation |

## 15. Future Scope

College survey dataset, additional UCI file (Mathematics course), merged-dataset analysis, Bayesian regression, mixed-effects models, calibration plots, exportable PDF report, multilingual UI.

## 16. Open Items (to resolve before Phase 2)

- Confirm from the UCI page: file names, instance/feature counts, license, citation text.
- Decide default performance bands (documented rationale needed).
- Confirm whether to treat G3 = 0 records separately (inspect before deciding).
