# EduLens: Requirements and implementation status

Priority: **M** = Must, **S** = Should, **C** = Could. Status reflects the Phase 10 source and
automated-test audit: **Done**, **Partial**, or **Not done**. “Done” records implementation and
available tests; it is not a claim of independent validation beyond those tests.

## Functional Requirements

| ID | Requirement | P | Status | Note |
|---|---|---:|---|---|
| FR-001 | Load the UCI Student Performance dataset from `data/raw/` (documented download step). | M | Done | Loads the supplied Portuguese or Mathematics CSV locally; fresh-setup source is documented. |
| FR-002 | Accept user CSV upload; validate encoding, delimiter, non-empty, minimum rows. | M | Done | Uploads are validated in memory; invalid-file behavior has AppTest coverage. |
| FR-003 | Let the user map columns to roles (target, numeric predictors, categorical predictors). | S | Partial | Upload schema is inspected; interactive user-defined role mapping is not implemented. |
| FR-004 | Detect missing values, duplicates, dtype problems; report counts. | M | Done | Dataset cleaning report preserves source rows and reports quality properties. |
| FR-005 | Inspect outliers (IQR rule) and report without silently removing them. | M | Done | IQR outlier counts are reported; no observations are silently removed. |
| FR-006 | Encode categorical variables where required, documenting the scheme. | M | Done | Modeling pipelines encode within the training pipeline; scheme is documented. |
| FR-007 | Show before/after cleaning statistics. | M | Done | Explorer displays the cleaning report and derived-column behavior. |
| FR-008 | Dataset Explorer: preview, search, filter, column info, download cleaned CSV. | M | Done | All listed explorer controls and the cleaned CSV download are implemented. |
| FR-009 | Descriptive stats for selectable variables (mean, median, mode, min, max, range, variance, SD, Q1–Q3, IQR, skewness). | M | Done | `src/descriptive_stats.py` implements selectable numeric/ordinal summaries. |
| FR-010 | EDA charts: histogram, KDE, box, violin, bar/count, scatter, grouped comparison. | M | Done | Figure builders are used by the dynamic EDA page; chart titles/captions are provided. |
| FR-011 | Preset relationship views: study time, absences, failures, G1/G2, family support, parental education, health, free time vs G3. | M | Partial | Builders exist in `src/visualization.py`; the page uses dynamic selections, not a dedicated preset selector. |
| FR-012 | Correlation matrix, heatmap and ranked association table for a selectable target. | M | Done | Page exposes target selection and both association summaries. |
| FR-013 | Probability calculator: P(A), P(B), P(A∩B), P(A\|B) with configurable threshold, condition variable and category. | M | Done | Event conditions, counts, Wilson interval and configured grade thresholds are used. |
| FR-014 | Bayes walkthrough: prior, likelihood, evidence, posterior from dataset frequencies. | M | Done | All four values and underlying sample counts are shown. |
| FR-015 | Distribution page: normal, binomial (suitable binary events), empirical; parameters and fit discussion. | S | Done | Distribution fits and goodness-of-fit evidence are available on the Probability page. |
| FR-016 | Hypothesis tests: independent t-test, one-way ANOVA with post-hoc, chi-square. | M | Done | All three test families and their implemented alternatives are exposed. |
| FR-017 | Confidence intervals for means, mean differences, proportions, regression coefficients; configurable level. | M | Done | Mean/test/OLS intervals use selected confidence setting; proportion intervals use Wilson. |
| FR-018 | Multiple linear regression with inference output and diagnostics plots. | M | Done | Coefficients and several diagnostics are exposed; see SR-013 for one missing plot. |
| FR-019 | Configurable Low/Medium/High thresholds with documented methodology. | M | Done | Sidebar cutoffs are shared with analysis and class-label construction. |
| FR-020 | Multinomial logistic regression giving class probabilities. | M | Done | Pipeline estimates and displays probabilities for all three classes. |
| FR-021 | Prediction page: input form, probability visualization, disclaimer. | M | Done | Inputs, probability chart, and required disclaimer are present. |
| FR-022 | Model evaluation page: confusion matrix, accuracy, precision, recall, F1, cross-validation. | M | Done | Holdout metrics and stratified CV summaries are displayed. |
| FR-023 | Model A vs Model B comparison view. | M | Done | Same split/folds are compared with observed differences and spread. |
| FR-024 | Sidebar: logo, dataset selector, upload, navigation, settings, About. | M | Done | Shared sidebar routes all 11 pages and exposes the shared settings. |
| FR-025 | About page: methods, ethics, limitations, dataset citation. | M | Done | About page contains all four requested topics and the DOI citation. |
| FR-026 | Export of tables (CSV) where useful. | C | Partial | Cleaned-dataset CSV is downloadable; arbitrary analysis tables lack general CSV export. |

## Non-Functional Requirements

| ID | Requirement | P | Status | Note |
|---|---|---:|---|---|
| NFR-001 | Self-contained Streamlit app; no external services. | M | Done | Runtime uses local files and no external service calls. |
| NFR-002 | Cache data loading (`st.cache_data`) and model fitting (`st.cache_resource`/`cache_data`). | M | Done | Data and expensive model-analysis helpers use Streamlit caching. |
| NFR-003 | Page interactions respond promptly on a typical laptop for the UCI-sized data (target to be measured, not assumed). | S | Not done | No formal responsiveness target or benchmark has been measured. |
| NFR-004 | Modular code; statistics in `src/`, no business logic in page files. | M | Done | Statistical functions are in `src/`; `views/_analysis.py` contains cached orchestration. |
| NFR-005 | Reproducible: fixed random seed, pinned-range dependencies. | M | Partial | Default seed and deterministic splits are tested; dependency lower bounds are not upper-pinned. |
| NFR-006 | No raw tracebacks shown to users; errors logged. | M | Done | Expected errors render friendly messages; invalid-upload routes are smoke-tested. |
| NFR-007 | Unit tests for statistical and probability functions with hand-checkable fixtures. | M | Done | Hand calculations, library cross-checks and real-data assertions are in pytest. |
| NFR-008 | Code style via a formatter/linter (e.g. ruff/black), configured in repo. | S | Done | Ruff and Black are configured and run in Phase 10 validation. |
| NFR-009 | Uploaded data is not persisted. | M | Done | Uploaded bytes stay in memory; upload path has no disk-write step. |
| NFR-010 | Works on current Python 3.x supported by all dependencies. | M | Partial | Python 3.11 is the project target and tested interpreter; other Python versions are unverified. |
| NFR-011 | Docs match implemented behavior; updated with each phase. | M | Done | Phase 10 updates README, method, data, architecture and requirement documentation. |

## Statistical Requirements

| ID | Requirement | P | Status | Note |
|---|---|---:|---|---|
| SR-001 | Never claim causation; use "association" / "model contribution" wording. | M | Done | UI and docs use observational/non-causal caveats; the required disclaimer is preserved. |
| SR-002 | Every test shows H0, H1, α, statistic, p-value, decision, interpretation. | M | Done | Test result panels render the ordered result flow and hypotheses/threshold. |
| SR-003 | Every test lists assumptions and runs feasible checks (Shapiro–Wilk, Levene, expected counts, VIF, etc.). | M | Done | Test results include relevant check objects and plain-English statuses. |
| SR-004 | On assumption violation: warn and offer alternative (Welch, Mann–Whitney, Kruskal–Wallis, Fisher, robust SE). | M | Partial | Alternatives are reported/suggested; selection is not automatically changed and some sparse-table actions remain user choices. |
| SR-005 | Report effect sizes (Cohen's d, eta-squared, Cramér's V) alongside p-values. | S | Done | Implemented test results expose the corresponding effect size. |
| SR-006 | ANOVA significant → Tukey HSD (or documented alternative). | M | Done | Tukey HSD runs when classical ANOVA is significant; omnibus-only Kruskal limitation is documented. |
| SR-007 | Multiple comparisons: state correction used or caution if none. | S | Partial | Tukey's within-test adjustment is shown; no global correction spans separate selected tests. |
| SR-008 | Pearson for linear numeric relationships; Spearman for ordinal/skewed; ordinal variables flagged. | M | Done | Auto-ranking recommends Spearman for configured ordinal variables and labels methods. |
| SR-009 | Conditional probabilities computed from empirical frequencies; show counts, not just percentages. | M | Done | Probability and Bayes displays report sample counts with empirical estimates. |
| SR-010 | Handle P(B)=0 and tiny cells with explicit messages. | M | Done | Zero conditioning events and very small denominators return explanatory messages. |
| SR-011 | Threshold methodology documented and displayed. | M | Done | Fixed bands and user-adjusted cutoffs are documented and shown in the UI. |
| SR-012 | Distribution fits not forced; report fit statistics and "does not fit well" outcomes. | M | Done | Fit verdicts include evidence; normality tests note approximate KS p-values. |
| SR-013 | Regression diagnostics: residual vs fitted, Q–Q, scale-location, leverage/Cook's, VIF. | M | Partial | Residual/fitted, Q–Q, leverage/Cook's and VIF exist; scale-location plot is not implemented. |
| SR-014 | Confidence intervals use the correct distribution (t for means; Wilson or exact for proportions). | M | Done | Means use t intervals; event proportions use Wilson intervals. |
| SR-015 | Every displayed statistic is computed, never hard-coded. | M | Done | UI result values come from source data and statistical/model functions; defaults are configuration parameters. |

## Prediction Requirements

| ID | Requirement | P | Status | Note |
|---|---|---:|---|---|
| PR-001 | Model A excludes G1 and G2; enforced by code guard plus automated test. | M | Done | Configured feature set and encoded Model A names are tested; UI controls exclude them. |
| PR-002 | Model B = Model A + G1 + G2. | M | Done | Feature construction and tests enforce the exact extension. |
| PR-003 | Stratified train/test split with fixed seed. | M | Done | Split is stratified and deterministic from the configured seed. |
| PR-004 | Scaling/encoding inside sklearn `Pipeline`, fit on training data only. | M | Done | Preprocessing is in the pipeline and leakage tests cover training-only fitting. |
| PR-005 | Cross-validation (stratified k-fold) reported with mean and spread. | M | Done | Fold metrics include mean and sample SD. |
| PR-006 | Metrics: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix. | M | Done | Holdout page reports the full requested metric set. |
| PR-007 | Predict probabilities for all three classes; they sum to 1. | M | Done | Probability outputs and normalization are tested. |
| PR-008 | Prediction UI shows disclaimer verbatim. | M | Done | The exact approved disclaimer is present in the Prediction view. |
| PR-009 | Explain model limits, class imbalance and unreliability of individual predictions. | M | Done | UI and About text qualify population limits and individual probabilities. |
| PR-010 | Model contribution view uses coefficients, with the caveat of non-causality. | M | Done | Coefficients/odds ratios display the model-contribution caveat. |
| PR-011 | Optional comparison against a baseline (majority class) for context. | S | Done | Majority-class accuracy and macro-F1 baselines are shown beside model metrics. |

## UI Requirements

| ID | Requirement | P | Status | Note |
|---|---|---:|---|---|
| UI-001 | CSS variables for background, surface, card, border, primary, secondary, text, muted, gradient, shadow, radius. | M | Done | Design tokens are defined in the EduLens stylesheet. |
| UI-002 | Dark midnight theme with indigo/blue/violet/cyan accents; avoid heavy neon. | M | Done | Shared theme uses the approved palette. |
| UI-003 | Hero and metric cards on Overview. | M | Done | Overview uses a hero and data-computed metric cards. |
| UI-004 | "How EduLens Works" pipeline graphic: DATA → STATISTICS → PROBABILITY → MODELLING → INSIGHTS. | M | Done | Overview contains the workflow strip. |
| UI-005 | Hypothesis page flow: TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION. | M | Done | Hypothesis results use the required ordered labels. |
| UI-006 | Decision indicators use neutral wording ("Evidence against H0" / "Insufficient evidence"), icon plus text, not color alone. | M | Done | Shared status component pairs icon and neutral text. |
| UI-007 | Large probability cards on Probability and Prediction pages. | M | Done | Probability calculator and profile predictions render labeled cards/charts. |
| UI-008 | Animations: fade-in, hover, counters (CSS), subtle gradient; respect `prefers-reduced-motion`. | M | Done | CSS includes subtle effects and a reduced-motion override. |
| UI-009 | Responsive layout; columns collapse on narrow screens. | M | Partial | Streamlit columns provide responsive layout, but narrow-screen behavior has not been separately tested. |
| UI-010 | Accessibility: contrast ≥ WCAG AA targets, readable font sizes, labelled controls, titled charts, alt text/captions. | M | Partial | Controls and chart labels/captions exist; contrast and screen-reader accessibility have not been formally audited. |
| UI-011 | Loading indicators for slow operations. | S | Done | Dataset and model operations use loading indicators/spinners. |
| UI-012 | Consistent plotting theme across Plotly/Matplotlib/Seaborn. | S | Partial | Shared plotting styles are applied, but full cross-library visual parity is not formally verified. |
