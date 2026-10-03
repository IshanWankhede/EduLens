"""Methods, ethics, limitations, and citation for EduLens."""

from __future__ import annotations

import streamlit as st

st.title("About EduLens")
st.markdown(
    "EduLens is a Probability & Statistics course project for exploring student-performance "
    "data through descriptive summaries, exploratory visualization, probability, inference, "
    "regression, and classification."
)

st.header("Methods")
st.markdown("""
- **Descriptive statistics:** sample summaries, group summaries with confidence intervals,
  frequencies, and explicitly reported missingness.
- **Exploration and association:** interactive distribution and comparison charts; Pearson
  and Spearman correlations with pairwise sample counts and uncertainty information.
- **Probability:** empirical conditional probabilities, Wilson intervals, a Bayes calculation
  with its intermediate counts, and distribution diagnostics.
- **Hypothesis testing:** Welch's independent t-test, one-way ANOVA with alternatives, and
  chi-square independence with expected-count checks.
- **Regression:** OLS with explicit encoding, confidence intervals, optional HC3 standard errors,
  residual diagnostics, and influence checks.
- **Classification:** leakage-safe, pipeline-based Low/Medium/High multinomial models. Model A
  excludes G1 and G2; Model B is separately labeled and includes those prior grades.
""")

st.header("Ethics and interpretation")
st.markdown("""
EduLens reports association, model contribution, and estimated probabilities. Statistical
relationships are not causal explanations, and a model output is not a guarantee about an
individual. Results depend on the observed data, selected variables, model assumptions, and
evaluation design. Group summaries can conceal variation within groups; small or imbalanced
groups warrant particular caution.

Student records are sensitive. Use uploads only when authorized, avoid including identifying
information, and follow applicable privacy requirements. Uploaded data is processed in memory
by the app's ingestion path and is not intentionally written to a dataset file.
""")

st.header("Limitations")
st.markdown("""
- The supplied student records are observational and may not represent other schools,
  populations, or time periods.
- Missing or excluded observations, measurement choices, and self-reported variables can affect
  results.
- Assumption checks are diagnostics rather than proof that assumptions hold.
- Correlations, confidence intervals, p-values, and predictive metrics each answer different
  questions; none alone establishes practical importance.
- Model A and Model B use different predictor information. Their comparison is specific to the
  displayed split and folds and should not be read as a universal ranking.
- Prediction uses fixed, configured G3 grade bands and can be uncertain, especially for groups
  with limited representation.
""")

st.header("Dataset citation")
st.markdown(
    "Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. "
    "[DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)."
)
