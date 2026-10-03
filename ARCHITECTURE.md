# EduLens: Architecture

## 1. Design Principles

1. **Statistics live in `src/`, never in page files.** Pages only collect inputs, call `src/` functions and render results. This makes every statistic unit-testable and traceable.
2. **Pure functions, typed results.** Statistical functions take a DataFrame and parameters and return a dataclass/dict (never print or call Streamlit).
3. **One source of truth for the data.** A single `DatasetBundle` (raw, cleaned, metadata, column roles) flows through the app via `st.session_state`.
4. **No leakage by construction.** Feature sets for Model A and Model B are defined in one place (`config.py`) and guarded by a test.
5. **Simple stack.** Streamlit + Python scientific stack. No backend services.

## 2. Changes to the suggested folder structure (and why)

| Change | Reason |
|---|---|
| `pages/` → **`views/`** | A folder named `pages/` makes Streamlit auto-generate its default sidebar navigation, which conflicts with the custom EduLens sidebar. We use `st.navigation` / a custom router in `app.py` with `views/`. |
| Added `src/config.py` | Central place for column roles, G1/G2 exclusion list, default thresholds, seed, α. |
| Added `src/ui/` (`theme.py`, `components.py`) | Injects CSS variables and reusable cards (metric card, probability card, status chip). Keeps CSS out of analysis code. |
| Added `src/validation.py` | CSV validation and user-friendly error types in one place. |
| Added `src/assumptions.py` | Shapiro, Levene, expected-count and VIF checks reused by tests and regression. |
| Added `assets/styles/edulens.css` | CSS lives in a file, loaded once. |
| Tests mirror `src/` | `tests/test_probability.py`, etc. |

## 3. Folder Structure

```
edulens/
├── app.py                      # entry: set_page_config, theme, sidebar, router
├── views/                      # one render() per page (UI only)
│   ├── overview.py
│   ├── dataset_explorer.py
│   ├── descriptive_statistics.py
│   ├── exploratory_analysis.py
│   ├── correlation.py
│   ├── probability.py
│   ├── hypothesis_testing.py
│   ├── regression.py
│   ├── prediction.py
│   ├── model_evaluation.py
│   └── about.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── descriptive_stats.py
│   ├── correlation.py
│   ├── probability.py          # conditional, Bayes, distributions
│   ├── assumptions.py
│   ├── hypothesis_tests.py
│   ├── regression.py
│   ├── prediction.py           # pipelines, Model A / B, metrics
│   ├── visualization.py
│   └── ui/
│       ├── theme.py
│       └── components.py
├── data/
│   ├── raw/                    # downloaded UCI files (not committed if policy says so)
│   ├── processed/
│   └── README.md
├── models/                     # generated artifacts (git-ignored)
├── assets/
│   ├── logo/
│   ├── images/
│   └── styles/edulens.css
├── tests/
├── docs/
├── requirements.txt
├── README.md  PRD.md  ARCHITECTURE.md  DATASET.md
├── STATISTICAL_METHODS.md  DESIGN.md  REQUIREMENTS.md
├── CONTRIBUTING.md  .gitignore
```

## 4. System Architecture

```mermaid
flowchart LR
    subgraph Sources
        U1[UCI Student Performance files]
        U2[User-uploaded CSV]
        U3[Future survey CSV]
    end
    subgraph Core["src/ (pure Python)"]
        DL[data_loader + validation]
        PP[preprocessing]
        DS[descriptive_stats]
        CO[correlation]
        PR[probability]
        HT[hypothesis_tests + assumptions]
        RG[regression]
        PD[prediction]
        VZ[visualization]
    end
    subgraph UI["Streamlit"]
        APP[app.py router + theme]
        V[views/*]
    end
    U1 --> DL
    U2 --> DL
    U3 --> DL
    DL --> PP
    PP --> DS & CO & PR & HT & RG & PD
    DS & CO & PR & HT & RG & PD --> VZ
    VZ --> V
    DS & CO & PR & HT & RG & PD --> V
    APP --> V
```

## 5. Component Architecture (Streamlit)

```mermaid
flowchart TB
    APP[app.py] --> TH[ui.theme: load CSS variables]
    APP --> SB[Sidebar: dataset selector, upload, navigation, settings]
    APP --> RT{Router}
    RT --> P1[Overview] & P2[Dataset Explorer] & P3[Descriptive Stats]
    RT --> P4[EDA] & P5[Correlation] & P6[Probability]
    RT --> P7[Hypothesis Testing] & P8[Regression]
    RT --> P9[Prediction] & P10[Model Evaluation] & P11[About]
    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9 & P10 --> CMP[ui.components: metric card, probability card, status chip, assumptions panel]
    SS[(st.session_state: DatasetBundle, settings)] --- SB
    SS --- RT
```

**Shared settings (sidebar → session state):** dataset choice, confidence level (default 0.95), α (default 0.05), performance threshold method and values, random seed.

## 6. Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Sidebar
    participant Loader as data_loader
    participant Val as validation
    participant Prep as preprocessing
    participant View as view page
    participant Stats as src module
    User->>Sidebar: choose UCI or upload CSV
    Sidebar->>Loader: load (cached)
    Loader->>Val: validate schema, dtypes, size
    Val-->>Sidebar: friendly errors if invalid
    Val->>Prep: clean + encode + derive performance category
    Prep-->>View: DatasetBundle (raw, clean, report)
    User->>View: select variables / parameters
    View->>Stats: call pure function
    Stats-->>View: result object (numbers, tables)
    View-->>User: charts, cards, interpretation
```

## 7. Statistical Pipeline

```mermaid
flowchart LR
    A[Clean data] --> B[Descriptive statistics]
    B --> C[Distributions & EDA]
    C --> D["Probability: P(A), P(A given B), Bayes"]
    D --> E[Hypothesis tests: t, ANOVA, chi-square]
    E --> F[Correlation: Pearson / Spearman]
    F --> G[Multiple linear regression + diagnostics]
    G --> H[Classification: class probabilities]
```

Each stage reads the cleaned data and the shared settings. No stage mutates the data.

## 8. Prediction Pipeline (leakage-safe)

```mermaid
flowchart TB
    D[Cleaned data] --> T[Create performance category from G3 using chosen thresholds]
    T --> S[Stratified train/test split, fixed seed]
    S --> FA[Model A features: no G1/G2]
    S --> FB[Model B features: A + G1 + G2]
    FA --> PA[Pipeline: impute → encode/scale → multinomial LogReg]
    FB --> PB[Pipeline: impute → encode/scale → multinomial LogReg]
    PA --> EA[Metrics + CV on test/folds]
    PB --> EB[Metrics + CV on test/folds]
    EA & EB --> CMP[Model A vs B comparison]
```

Rules:
- The category is derived from **G3**; G3 is never a feature.
- Preprocessing is inside the `Pipeline`, so it is fit on training folds only.
- `config.LEAKY_COLUMNS = ["G1", "G2", "G3"]` is removed for Model A; a unit test asserts none appear in its feature names (after encoding too).
- Thresholds derived from quantiles are computed on the **training** set only, or documented as a fixed-band choice to avoid leakage.

## 9. Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Constants: seed, α, column roles, leaky columns, default bands |
| `data_loader.py` | Load UCI files / uploaded CSV; cached; returns raw DataFrame + metadata |
| `validation.py` | Encoding/delimiter, empty data, required columns, dtype checks → friendly exceptions |
| `preprocessing.py` | Missing/duplicate/outlier reports, encoding, derived categories, before/after report |
| `descriptive_stats.py` | Summary table, mode, quartiles, IQR, skewness, group summaries |
| `correlation.py` | Pearson/Spearman matrices, ranked table, strength labels |
| `probability.py` | P(A), P(B), P(A∩B), P(A\|B), Bayes, binomial/normal/empirical fits |
| `assumptions.py` | Shapiro–Wilk, Levene, expected counts, VIF, residual tests |
| `hypothesis_tests.py` | t-test (Welch default), ANOVA + Tukey, chi-square, effect sizes, CIs |
| `regression.py` | statsmodels OLS, diagnostics data, coefficient tables |
| `prediction.py` | Pipelines, split, CV, metrics, Model A/B, probability output |
| `visualization.py` | Plotly/Matplotlib/Seaborn figure builders with shared theme |
| `ui/theme.py` | Inject CSS, define palette tokens |
| `ui/components.py` | HTML/markdown cards and chips (escape user text) |
| `views/*` | Layout and widgets only |

## 10. Dependency Flow

`views → ui.components, visualization, src modules → config`; `src` modules never import `views` or `streamlit` (except a thin caching wrapper layer in `data_loader`/`prediction`, or caching applied in views).

```mermaid
flowchart TB
    views --> src_stats[src statistical modules]
    views --> visualization
    views --> ui
    src_stats --> preprocessing --> validation --> data_loader
    src_stats --> assumptions
    src_stats --> config
    visualization --> config
```

## 11. Streamlit Architecture Notes

- **Routing:** `app.py` renders sidebar, reads the selected page, calls `views.<page>.render(bundle, settings)`.
- **Caching:** `st.cache_data` for loading, cleaning and deterministic statistics (keyed on data hash and parameters); `st.cache_resource` for fitted model objects.
- **State:** `st.session_state` holds the active `DatasetBundle`, uploaded-file hash and settings; changing the dataset invalidates dependent caches.
- **CSS:** one stylesheet injected once via `st.markdown(unsafe_allow_html=True)`; variables on `:root`; Streamlit's internal class names are targeted sparingly, because they can change between versions.
- **Error handling:** custom exceptions (`DataValidationError`, `InsufficientDataError`, `AssumptionWarning`) caught in a decorator that shows `st.error`/`st.warning` and logs details.
- **Security:** user-provided text (column names, categories) is HTML-escaped before inclusion in custom HTML.

## 12. Testing Strategy

- Unit tests with tiny hand-calculable fixtures (probabilities, Bayes, chi-square).
- Cross-check against SciPy/statsmodels results.
- Leakage test for Model A feature names.
- Validation tests for malformed CSVs.
- Streamlit smoke test via `streamlit.testing.v1.AppTest` (verify availability for the pinned version).
