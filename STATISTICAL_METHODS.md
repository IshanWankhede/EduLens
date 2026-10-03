# EduLens: Statistical Methods

Each method lists: **Purpose · Formula · Assumptions · Variables · Example interpretation · Implementation.**
Example interpretations are **templates**. Real values are produced by the app from the data; none are written here.

Notation: n = sample size, x̄ = sample mean, s = sample SD, α = significance level (default 0.05), CL = confidence level (default 95%).

---

## Part I: Descriptive Statistics

### 1. Mean
- **Purpose:** Central tendency (balance point).
- **Formula:** x̄ = (1/n) Σ xᵢ
- **Assumptions:** Numeric (interval/ratio) data; sensitive to outliers and skew.
- **Variables:** G3, G1, G2, absences, age.
- **Interpretation:** "The average final grade in this sample is x̄ out of 20."
- **Implementation:** `pandas.Series.mean()`; NaNs excluded and counted.

### 2. Median
- **Purpose:** Robust central value.
- **Formula:** Middle value of sorted data (average of two middle values if n even).
- **Assumptions:** Ordinal or numeric.
- **Variables:** G3, absences.
- **Interpretation:** "Half the students scored at or below the median." A mean far from the median suggests skew.
- **Implementation:** `Series.median()`.

### 3. Mode
- **Purpose:** Most frequent value (main summary for categorical data).
- **Formula:** argmax of frequency.
- **Assumptions:** None; may be multiple modes.
- **Variables:** studytime, failures, categorical variables.
- **Interpretation:** "The most common weekly study-time category is…"
- **Implementation:** `Series.mode()`; list all ties.

### 4. Variance
- **Purpose:** Spread around the mean.
- **Formula:** s² = Σ(xᵢ − x̄)² / (n − 1) (sample variance)
- **Assumptions:** Numeric; sensitive to outliers.
- **Variables:** numeric columns.
- **Interpretation:** Larger variance = more spread; units are squared.
- **Implementation:** `Series.var(ddof=1)`.

### 5. Standard Deviation
- **Purpose:** Spread in original units.
- **Formula:** s = √s²
- **Assumptions:** As variance. Interpretation using "68–95–99.7" only if roughly normal.
- **Interpretation:** "Typical distance of a grade from the mean is about s points."
- **Implementation:** `Series.std(ddof=1)`.

### 6. Quartiles
- **Purpose:** Positional summary (Q1 = 25th, Q2 = median, Q3 = 75th percentile).
- **Formula:** Linear-interpolation percentile.
- **Assumptions:** Ordinal or numeric.
- **Interpretation:** "25% of students are at or below Q1."
- **Implementation:** `Series.quantile([.25,.5,.75], interpolation="linear")` (pandas default,
  explicitly selected).

### 7. IQR (and outlier rule)
- **Purpose:** Robust spread; outlier flagging.
- **Formula:** IQR = Q3 − Q1; flagged if x < Q1 − 1.5·IQR or x > Q3 + 1.5·IQR.
- **Assumptions:** None strict; the rule is a convention.
- **Interpretation:** "Values flagged as potential outliers are inspected, not automatically removed."
- **Implementation:** Vectorized masks; reports count and values.

### 8. Skewness
- **Purpose:** Asymmetry of distribution.
- **Formula:** Adjusted Fisher–Pearson skewness
  G₁ = √(n(n−1))/(n−2) · m₃/m₂^(3/2), where m₂ and m₃ are the second and third central moments
  using denominator n. This is the bias-corrected coefficient returned by pandas `Series.skew()`.
- **Assumptions:** Numeric; unstable for small n.
- **Interpretation:** Positive → long right tail; negative → long left tail; near 0 → roughly symmetric. Rules of thumb are guidance only.
- **Implementation:** `Series.skew()`; equivalent to `scipy.stats.skew(..., bias=False)`.

---

## Part II: Association

### 9. Pearson Correlation
- **Purpose:** Strength and direction of *linear* association between two numeric variables.
- **Formula:** r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √[Σ(xᵢ − x̄)² Σ(yᵢ − ȳ)²]; test statistic t = r√(n−2)/√(1−r²) with n−2 df.
- **Assumptions:** Numeric, linear relationship, no extreme outliers, approximate bivariate normality for the p-value, independent observations.
- **Variables:** G1/G2/G3, absences, age and other numeric pairs.
- **Interpretation:** "r indicates a [weak/moderate/strong] [positive/negative] linear association; this does not show that one variable causes the other." Strength labels use the rule of thumb |r| < 0.3 weak, 0.3 ≤ |r| ≤ 0.5 moderate, and |r| > 0.5 strong; these are conventions, displayed with the label.
- **Implementation:** `scipy.stats.pearsonr` with CI via Fisher z-transform.

### 10. Spearman Correlation
- **Purpose:** Monotonic association; suited to ordinal data (studytime, health, Medu) and skewed variables.
- **Formula:** Pearson correlation of ranks; with no ties ρ = 1 − 6Σdᵢ² / [n(n²−1)].
- **Assumptions:** Ordinal or numeric, monotonic relationship, independent observations; heavy ties reduce precision.
- **Interpretation:** "ρ describes how consistently higher values of X go with higher/lower values of Y."
- **Implementation:** `scipy.stats.spearmanr`; used by default for ordinal columns.

---

## Part III: Probability

### 11. Conditional Probability
- **Purpose:** Probability of event A given event B, estimated from sample frequencies.
- **Formula:** P(A|B) = P(A∩B) / P(B) = n(A∩B) / n(B), requires P(B) > 0.
- **Assumptions:** Rows are a random-like sample of the population of interest; empirical relative frequencies approximate probabilities. Small n(B) gives unstable estimates.
- **Variables:** A = "High performance" (G3 ≥ configurable threshold); B = studytime group, absence group, failures = 0, high previous grade, etc.
- **Interpretation:** "Among the n(B) students in condition B, n(A∩B) were high performers, so the estimated P(A|B) = …". Compared with P(A) to describe association (not causation). Independence check: A and B are independent if P(A|B) = P(A).
- **Implementation:** Counts first, then ratios; Wilson CI for proportions; explicit message when n(B) = 0 or very small.
- **Small-condition rule:** "Very small" means n(B) < 5, configured as `SMALL_CONDITIONAL_SAMPLE_SIZE` in `src/config.py`; this is a warning about instability, not a prohibition on calculating the empirical estimate.

### 12. Bayes' Theorem
- **Purpose:** Reverse conditioning, e.g., from P(High | study time) to P(study time | High).
- **Formula:** P(B|A) = P(A|B)·P(B) / P(A), where P(A) = P(A|B)P(B) + P(A|Bᶜ)P(Bᶜ) (law of total probability).
- **Assumptions:** Same as conditional probability; B and Bᶜ partition the sample.
- **Display:** Prior P(B) → Likelihood P(A|B) → Evidence P(A) → Posterior P(B|A), with the counts behind each step, plus a check against the direct count n(A∩B)/n(A).
- **Interpretation:** "Among high performers, the share who report high study time is …; this is different from the share of high-study-time students who are high performers."
- **Implementation:** Pure function returning all intermediate quantities; test asserts Bayes result equals direct frequency.

### 13. Distributions (Normal, Binomial, Empirical)
- **Purpose:** Describe variable shape and model suitable events.
- **Formulas:**
  - Normal: f(x) = (1/σ√2π) exp(−(x−μ)²/2σ²), parameters μ̂ = x̄, σ̂ = s.
  - Binomial: P(X=k) = C(n,k) pᵏ(1−p)ⁿ⁻ᵏ, p̂ = proportion for a binary event (e.g., "High performer").
  - Empirical: ECDF F̂(x) = (1/n) Σ 1{xᵢ ≤ x}.
- **Assumptions:** Normal: continuous, roughly symmetric (grades are bounded integers 0–20, so normality is only approximate). Binomial: fixed n, independent trials, constant p.
- **Goodness of fit:** Q–Q plot, Shapiro–Wilk (note: very sensitive at larger n), Kolmogorov–Smirnov (parameters estimated from data → p-values approximate; state this), overlay of fitted vs empirical.
- **Interpretation:** Report "reasonable fit" or "poor fit" honestly. Do not force a fit. Binomial use: "Number of high performers in a random group of k students with p̂ = …", assuming independence.
- **Implementation:** `scipy.stats.norm`, `binom`, `shapiro`, `kstest`; plots in `visualization.py`.

---

## Part IV: Inference

### 14. Confidence Intervals
- **Purpose:** Range of plausible values for a parameter at CL.
- **Formulas:**
  - Mean: x̄ ± t_{α/2, n−1} · s/√n
  - Difference of means (Welch): (x̄₁ − x̄₂) ± t_{α/2, ν} √(s₁²/n₁ + s₂²/n₂), ν by Welch–Satterthwaite
  - Proportion (Wilson): (p̂ + z²/2n ± z√[p̂(1−p̂)/n + z²/4n²]) / (1 + z²/n)
  - Regression coefficient: β̂ ± t_{α/2, n−p−1} · SE(β̂)
- **Assumptions:** Independent observations; approximately normal sampling distribution (CLT for moderate/large n); Wilson works better than Wald for small n or extreme p.
- **Interpretation:** "If we repeated the sampling many times, about CL% of intervals built this way would contain the true value." It is **not** "95% probability the true value is in this interval."
- **Implementation:** `scipy.stats.t`, `statsmodels.stats.proportion.proportion_confint(method="wilson")`; confidence level from settings.

### 15. Independent Samples t-Test
- **Purpose:** Compare means of two independent groups (e.g., high vs low study time).
- **Hypotheses:** H₀: μ₁ = μ₂; H₁: μ₁ ≠ μ₂ (two-sided by default).
- **Formula (Welch, default):** t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂); df via Welch–Satterthwaite. Student's pooled version available when equal variances are justified.
- **Assumptions:** Independence between and within groups; interval-scale outcome (grades treated as approximately interval); approximate normality of group means or large n; variances not assumed equal under Welch.
- **Checks:** Shapiro–Wilk per group, Levene's test, group sizes, histogram/Q–Q.
- **Effect size:** Cohen's d (with Hedges' correction for small n).
- **Alternative if violated:** Mann–Whitney U.
- **Interpretation template:** "At α = 0.05, there is [sufficient / insufficient] evidence to conclude that mean G3 differs between groups (t = …, df = …, p = …, 95% CI for difference [… , …]). The groups differ by d = … SD. This shows association, not that study time causes the difference."
- **Implementation:** `scipy.stats.ttest_ind(equal_var=False)`; CI computed explicitly.
- **Note:** Splitting a numeric/ordinal variable into groups (e.g., cutoffs of studytime) is a choice that must be shown on screen.

### 16. One-Way ANOVA (+ post-hoc)
- **Purpose:** Compare means across ≥3 groups (e.g., studytime levels 1–4).
- **Hypotheses:** H₀: all group means equal; H₁: at least one differs.
- **Formula:** F = MS_between / MS_within = [Σnⱼ(x̄ⱼ − x̄)²/(k−1)] / [Σ(xᵢⱼ − x̄ⱼ)²/(N−k)]
- **Assumptions:** Independence; approximately normal residuals within groups; homogeneity of variance.
- **Checks:** Shapiro–Wilk on residuals, Levene's test, group sizes.
- **Effect size:** η² = SS_between / SS_total.
- **Alternatives if violated:** Welch's ANOVA (unequal variances), Kruskal–Wallis (non-normal/ordinal).
- **Post-hoc:** Tukey HSD if classical ANOVA is significant. This implementation does not provide Dunn pairwise testing: Kruskal–Wallis is reported as a nonparametric omnibus alternative only, so pairwise nonparametric follow-up requires a separately chosen and documented multiplicity-corrected method.
- **Interpretation template:** "F(df₁, df₂) = …, p = …. [Sufficient/Insufficient] evidence that not all group means are equal. Post-hoc comparisons indicate which pairs differ (adjusted p-values)." Group means shown in a table with CIs.
- **Implementation:** `scipy.stats.f_oneway`, `statsmodels` `anova_lm`, `pairwise_tukeyhsd`.

### 17. Chi-Square Test of Independence
- **Purpose:** Test association between two categorical variables (e.g., study-time category vs performance category).
- **Hypotheses:** H₀: variables independent; H₁: associated.
- **Formula:** χ² = Σ (Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ with Eᵢⱼ = (row total × column total) / N; df = (r − 1)(c − 1).
- **Assumptions:** Both variables categorical; independent observations; expected counts adequate (common guideline: all Eᵢⱼ ≥ 1 and at least ~80% ≥ 5).
- **Display:** Observed table, expected table, standardized residuals, χ², df, p, Cramér's V = √(χ² / (N·min(r−1, c−1))).
- **Alternatives if violated:** Merge sparse categories (documented), Fisher's exact test for 2×2, Monte Carlo/permutation p-value.
- **Interpretation template:** "χ²(df) = …, p = …. [Sufficient/Insufficient] evidence of an association between X and Y (Cramér's V = …). This does not indicate direction or causation."
- **Implementation:** `scipy.stats.chi2_contingency` (note: Yates correction on 2×2 by default; state setting).

**Decision language (all tests):** "Sufficient evidence against H₀" / "Insufficient evidence against H₀". We do not say "accept H₀" or "prove". p-values are not the probability that H₀ is true. Multiple tests increase false-positive risk; if many tests are run, this is stated.

---

## Part V: Models

### 18. Multiple Linear Regression (OLS)
- **Purpose:** Estimate the association between G3 and several predictors simultaneously, holding other included variables constant.
- **Formula:** y = β₀ + β₁x₁ + … + β_p x_p + ε; β̂ = (XᵀX)⁻¹Xᵀy.
- **Output:** Coefficients, SE, t, p, CI, R², adjusted R² = 1 − (1−R²)(n−1)/(n−p−1), F-test, residual SE.
- **Assumptions:** Linearity; independent errors; homoscedasticity; approximately normal residuals (for small-sample inference); no severe multicollinearity; no highly influential points.
- **Diagnostics:** Residuals vs fitted, Q–Q plot, scale–location, leverage/Cook's distance, VIF (concern flagged using a documented guideline, e.g., VIF > 5 or 10), Breusch–Pagan, Durbin–Watson (limited relevance for non-time data; noted).
- **Alternatives if violated:** Robust (HC) standard errors, transformations, removing/combining collinear predictors, ordinal/nonparametric methods.
- **Coding:** Binary → 0/1 indicators; nominal → dummy with reference level shown; ordinal predictors treated as numeric or categorical (choice displayed).
- **Variables:** Model A predictors exclude G1/G2. A Model B regression (with G1/G2) may be shown for comparison, labeled clearly.
- **Interpretation template:** "Holding the other included variables constant, each one-step increase in studytime is associated with a β̂-point change in G3 (95% CI […, …], p = …)." Never "studytime raises grades by".
- **Implementation:** `statsmodels.api.OLS` / formula API; diagnostics via `statsmodels.stats`.

### 19. Multinomial Logistic Regression
- **Purpose:** Estimate P(Low), P(Medium), P(High).
- **Formula (softmax):** P(y = k | x) = exp(β_kᵀx) / Σⱼ exp(β_jᵀx); coefficients fitted by (regularized) maximum likelihood. Coefficient exp(β) = odds ratio relative to the reference class for a one-unit change.
- **Assumptions:** Independent observations; correct specification (log-odds linear in predictors); no severe multicollinearity; adequate sample per class; categories mutually exclusive.
- **Variables:** Model A features; Model B = A + G1 + G2. The target category is derived from G3 using documented thresholds.
- **Imbalance handling:** Report class counts; consider `class_weight` only if justified; report macro-F1 and a majority-class baseline.
- **Interpretation:** "The model estimates a p% probability of the High category for this profile. This is an estimate based on the dataset, not a guarantee."
- **Implementation:** scikit-learn `LogisticRegression` inside `Pipeline`; statsmodels `MNLogit` optionally for coefficient p-values/CIs **[VERIFY convergence with the encoded features]**. Regularization settings are stated (scikit-learn applies L2 by default; this affects coefficient interpretation and p-values).

### 20. Model Evaluation Metrics
- **Purpose:** Measure how well predicted classes match the actual ones on unseen data.
- **Formulas (per class k, one-vs-rest):**
  - Accuracy = correct / total
  - Precision_k = TP_k / (TP_k + FP_k)
  - Recall_k = TP_k / (TP_k + FN_k)
  - F1_k = 2·Precision_k·Recall_k / (Precision_k + Recall_k)
  - Macro-F1 = mean of F1_k
- **Probability quality (optional):** log-loss, Brier score, calibration plot.
- **Methodology:** Stratified train/test split (fixed seed), stratified k-fold CV (mean ± SD), pipeline-embedded preprocessing, threshold computation not leaking test data. Compare to majority-class baseline.
- **Assumptions:** Test data representative of the future use case; independent test rows. With few students per class, metrics are noisy: show CV spread.
- **Model A vs B:** Same split and folds for both; report differences with spread. The expected larger benefit from G1/G2 (as noted by UCI) is shown only if the data demonstrates it.
- **Interpretation:** "Accuracy alone can mislead when classes are imbalanced; see per-class recall and macro-F1."
- **Implementation:** `sklearn.metrics`, `cross_validate`.

---

## Part VI: Performance Category Methodology

1. Categories derive from G3 only.
2. Thresholds are user-configurable (Low / Medium / High boundaries).
3. The active method (fixed bands or quantiles) and the exact numbers are displayed on every page that uses categories.
4. If quantiles are used for modeling, they are computed from training data only, or the choice is documented as a descriptive convention.
5. Sensitivity: the app lets users change thresholds and see the effect on probabilities and tests.

## Part VII: Cross-Cutting Rules

- All numbers shown in the UI are computed live from the loaded data.
- Every test panel includes assumptions, checks and alternatives.
- Terminology: "association", "model contribution", "estimated probability"; never "causes".
- Missing data: reported; rows dropped or imputed only with a visible, documented method.
- Reproducibility: fixed random seed in `config.py`; library versions recorded.
