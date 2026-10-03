# EduLens: Requirements

Priority: **M** = Must, **S** = Should, **C** = Could.

## Functional Requirements

| ID | Requirement | P |
|---|---|---|
| FR-001 | Load the UCI Student Performance dataset from `data/raw/` (documented download step). | M |
| FR-002 | Accept user CSV upload; validate encoding, delimiter, non-empty, minimum rows. | M |
| FR-003 | Let the user map columns to roles (target, numeric predictors, categorical predictors). | S |
| FR-004 | Detect missing values, duplicates, dtype problems; report counts. | M |
| FR-005 | Inspect outliers (IQR rule) and report without silently removing them. | M |
| FR-006 | Encode categorical variables where required, documenting the scheme. | M |
| FR-007 | Show before/after cleaning statistics. | M |
| FR-008 | Dataset Explorer: preview, search, filter, column info, download cleaned CSV. | M |
| FR-009 | Descriptive stats for selectable variables (mean, median, mode, min, max, range, variance, SD, Q1–Q3, IQR, skewness). | M |
| FR-010 | EDA charts: histogram, KDE, box, violin, bar/count, scatter, grouped comparison. | M |
| FR-011 | Preset relationship views: study time, absences, failures, G1/G2, family support, parental education, health, free time vs G3. | M |
| FR-012 | Correlation matrix, heatmap and ranked association table for a selectable target. | M |
| FR-013 | Probability calculator: P(A), P(B), P(A∩B), P(A\|B) with configurable threshold, condition variable and category. | M |
| FR-014 | Bayes walkthrough: prior, likelihood, evidence, posterior from dataset frequencies. | M |
| FR-015 | Distribution page: normal, binomial (suitable binary events), empirical; parameters and fit discussion. | S |
| FR-016 | Hypothesis tests: independent t-test, one-way ANOVA with post-hoc, chi-square. | M |
| FR-017 | Confidence intervals for means, mean differences, proportions, regression coefficients; configurable level. | M |
| FR-018 | Multiple linear regression with inference output and diagnostics plots. | M |
| FR-019 | Configurable Low/Medium/High thresholds with documented methodology. | M |
| FR-020 | Multinomial logistic regression giving class probabilities. | M |
| FR-021 | Prediction page: input form, probability visualization, disclaimer. | M |
| FR-022 | Model evaluation page: confusion matrix, accuracy, precision, recall, F1, cross-validation. | M |
| FR-023 | Model A vs Model B comparison view. | M |
| FR-024 | Sidebar: logo, dataset selector, upload, navigation, settings, About. | M |
| FR-025 | About page: methods, ethics, limitations, dataset citation. | M |
| FR-026 | Export of tables (CSV) where useful. | C |

## Non-Functional Requirements

| ID | Requirement | P |
|---|---|---|
| NFR-001 | Self-contained Streamlit app; no external services. | M |
| NFR-002 | Cache data loading (`st.cache_data`) and model fitting (`st.cache_resource`/`cache_data`). | M |
| NFR-003 | Page interactions respond promptly on a typical laptop for the UCI-sized data (target to be measured, not assumed). | S |
| NFR-004 | Modular code; statistics in `src/`, no business logic in page files. | M |
| NFR-005 | Reproducible: fixed random seed, pinned-range dependencies. | M |
| NFR-006 | No raw tracebacks shown to users; errors logged. | M |
| NFR-007 | Unit tests for statistical and probability functions with hand-checkable fixtures. | M |
| NFR-008 | Code style via a formatter/linter (e.g., ruff/black), configured in repo. | S |
| NFR-009 | Uploaded data is not persisted. | M |
| NFR-010 | Works on current Python 3.x supported by all dependencies. | M |
| NFR-011 | Docs match implemented behavior; updated with each phase. | M |

## Statistical Requirements

| ID | Requirement | P |
|---|---|---|
| SR-001 | Never claim causation; use "association" / "model contribution" wording. | M |
| SR-002 | Every test shows H0, H1, α, statistic, p-value, decision, interpretation. | M |
| SR-003 | Every test lists assumptions and runs feasible checks (Shapiro–Wilk, Levene, expected counts, VIF, etc.). | M |
| SR-004 | On assumption violation: warn and offer alternative (Welch, Mann–Whitney, Kruskal–Wallis, Fisher, robust SE). | M |
| SR-005 | Report effect sizes (Cohen's d, eta-squared, Cramér's V) alongside p-values. | S |
| SR-006 | ANOVA significant → Tukey HSD (or documented alternative). | M |
| SR-007 | Multiple comparisons: state correction used or caution if none. | S |
| SR-008 | Pearson for linear numeric relationships; Spearman for ordinal/skewed; ordinal variables flagged. | M |
| SR-009 | Conditional probabilities computed from empirical frequencies; show counts, not just percentages. | M |
| SR-010 | Handle P(B)=0 and tiny cells with explicit messages. | M |
| SR-011 | Threshold methodology documented and displayed. | M |
| SR-012 | Distribution fits not forced; report fit statistics and "does not fit well" outcomes. | M |
| SR-013 | Regression diagnostics: residual vs fitted, Q–Q, scale-location, leverage/Cook's, VIF. | M |
| SR-014 | Confidence intervals use the correct distribution (t for means; Wilson or exact for proportions). | M |
| SR-015 | Every displayed statistic is computed, never hard-coded. | M |

## Prediction Requirements

| ID | Requirement | P |
|---|---|---|
| PR-001 | Model A excludes G1 and G2; enforced by code guard plus automated test. | M |
| PR-002 | Model B = Model A + G1 + G2. | M |
| PR-003 | Stratified train/test split with fixed seed. | M |
| PR-004 | Scaling/encoding inside sklearn `Pipeline`, fit on training data only. | M |
| PR-005 | Cross-validation (stratified k-fold) reported with mean and spread. | M |
| PR-006 | Metrics: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix. | M |
| PR-007 | Predict probabilities for all three classes; they sum to 1. | M |
| PR-008 | Prediction UI shows disclaimer verbatim. | M |
| PR-009 | Explain model limits, class imbalance and unreliability of individual predictions. | M |
| PR-010 | Model contribution view uses coefficients, with the caveat of non-causality. | M |
| PR-011 | Optional comparison against a baseline (majority class) for context. | S |

## UI Requirements

| ID | Requirement | P |
|---|---|---|
| UI-001 | CSS variables for background, surface, card, border, primary, secondary, text, muted, gradient, shadow, radius. | M |
| UI-002 | Dark midnight theme with indigo/blue/violet/cyan accents; avoid heavy neon. | M |
| UI-003 | Hero and metric cards on Overview. | M |
| UI-004 | "How EduLens Works" pipeline graphic: DATA → STATISTICS → PROBABILITY → MODELLING → INSIGHTS. | M |
| UI-005 | Hypothesis page flow: TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION. | M |
| UI-006 | Decision indicators use neutral wording ("Evidence against H0" / "Insufficient evidence"), icon plus text, not color alone. | M |
| UI-007 | Large probability cards on Probability and Prediction pages. | M |
| UI-008 | Animations: fade-in, hover, counters (CSS), subtle gradient; respect `prefers-reduced-motion`. | M |
| UI-009 | Responsive layout; columns collapse on narrow screens. | M |
| UI-010 | Accessibility: contrast ≥ WCAG AA targets, readable font sizes, labelled controls, titled charts, alt text/captions. | M |
| UI-011 | Loading indicators for slow operations. | S |
| UI-012 | Consistent plotting theme across Plotly/Matplotlib/Seaborn. | S |
