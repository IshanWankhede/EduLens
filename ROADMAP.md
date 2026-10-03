# EduLens: Development Roadmap

Each phase ends with: tests passing, docs updated, and your approval before the next phase.

| Phase | Scope | Deliverables | Exit criteria |
|---|---|---|---|
| 1 | Documentation + architecture | PRD, REQUIREMENTS, ARCHITECTURE, DATASET, STATISTICAL_METHODS, DESIGN, README, requirements, .gitignore | You approve the docs |
| 2 | Ingestion + cleaning | `config`, `data_loader`, `validation`, `preprocessing`; verify [VERIFY] dataset facts; finalize `DATASET.md` and `data/README.md` | Loads UCI + CSV; cleaning report; G3=0 and overlap decisions recorded |
| 3 | Descriptive statistics | `descriptive_stats`, tests with hand-checked fixtures | Table matches pandas/SciPy |
| 4 | EDA + visualization | `visualization`, chart builders, preset relationship views | All required charts titled and captioned |
| 5 | Probability | `probability` (conditional, Bayes, distributions) | Bayes equals direct frequency in tests; edge cases handled |
| 6 | Hypothesis testing | `assumptions`, `hypothesis_tests`, effect sizes, post-hoc | Each test cross-checked with SciPy/statsmodels |
| 7 | Regression | `regression`, diagnostics | Output matches statsmodels reference; assumptions panel works |
| 8 | Prediction | `prediction`: Model A/B pipelines, CV, metrics | Leakage test passes; A vs B comparison reproducible |
| 9 | Streamlit UI | `app.py`, `views/*`, `ui/*`, CSS, `.streamlit/config.toml` | All 11 pages work; accessibility checklist done |
| 10 | Testing + documentation | Full test run, AppTest smoke tests, docs aligned with behavior, real screenshots | README placeholders replaced with real content |
| 11 | Deployment preparation | Dependency freeze, run instructions validated on a clean environment, optional Streamlit Community Cloud notes | Fresh-clone run succeeds |

## Open Decisions (before Phase 2)

1. Performance bands: fixed vs quantile (see DATASET §5).
2. Primary file: Portuguese only, or both courses analyzed separately.
3. License for the project code.
4. Python version target.
