# EduLens: Dataset

All facts below come from the official UCI page, retrieved while preparing this document. Items marked **[VERIFY]** are not stated on that page and must be confirmed after downloading the files (Phase 2); they are not assumed.

## 1. Identification

| Item | Value |
|---|---|
| Name | Student Performance |
| Repository | UCI Machine Learning Repository |
| Official page | https://archive.ics.uci.edu/dataset/320/student+performance |
| Download | https://archive.ics.uci.edu/static/public/320/student+performance.zip |
| DOI | https://doi.org/10.24432/C5TG7T |
| Donated | 11/26/2014 |
| Creator | Paulo Cortez |
| Subject area | Social Science |
| Associated tasks | Classification, Regression |
| Feature type | Integer (as listed by UCI) |
| Instances (as listed) | 649 |
| Features (as listed) | 30 (plus G1, G2, G3 grades) |
| Missing values | "No" per UCI |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0): sharing and adaptation allowed with appropriate credit |

**Recommended citation (from UCI):**
Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

**Introductory paper (from UCI):** P. Cortez and A. M. G. Silva, "Using data mining to predict secondary school student performance," Proceedings of the 5th Annual Future Business Technology Conference, 2008.

## 2. Files and Population

UCI describes data from **secondary education at two Portuguese schools** (Gabriel Pereira "GP", Mousinho da Silveira "MS"), collected from school reports and questionnaires. Two datasets are provided, one per subject:

- `student-mat.csv`: Mathematics course
- `student-por.csv`: Portuguese language course

**[VERIFY]** Row counts per file, delimiter (expected to be semicolon-separated, to be confirmed), and the exact file contents inside `student.zip`. The 649 total listed by UCI should be matched with the row count of the file we load. EduLens v1 default: the Portuguese file; the Mathematics file is optional. The two files are **not** treated as one combined sample unless overlap between students is investigated and documented **[VERIFY whether students overlap and how it is identifiable]**.

## 3. Variables (as documented by UCI)

| # | Variable | Description (UCI wording condensed) | Type | EduLens role |
|---|---|---|---|---|
| 1 | school | GP or MS | Categorical (binary) | grouping / context |
| 2 | sex | F or M | Binary | demographic (sensitive) |
| 3 | age | 15 to 22 | Integer | demographic |
| 4 | address | U urban, R rural | Binary | demographic |
| 5 | famsize | LE3 (≤3) or GT3 (>3) | Binary | family |
| 6 | Pstatus | T together, A apart | Binary | family |
| 7 | Medu | Mother's education 0 none, 1 primary (4th grade), 2 5th–9th grade, 3 secondary, 4 higher | Ordinal (0–4) | predictor |
| 8 | Fedu | Father's education, same scale | Ordinal (0–4) | predictor |
| 9 | Mjob | teacher, health, services, at_home, other | Nominal | family |
| 10 | Fjob | same categories | Nominal | family |
| 11 | reason | Reason for choosing school: home, reputation, course, other | Nominal | context |
| 12 | guardian | mother, father, other | Nominal | family |
| 13 | traveltime | 1 <15 min, 2 15–30 min, 3 30 min–1 h, 4 >1 h | Ordinal | predictor |
| 14 | studytime | Weekly study time: 1 <2 h, 2 2–5 h, 3 5–10 h, 4 >10 h | Ordinal | key predictor |
| 15 | failures | Past class failures (UCI: "n if 1<=n<3, else 4") | Ordinal/count | key predictor |
| 16 | schoolsup | Extra educational support | Binary | predictor |
| 17 | famsup | Family educational support | Binary | predictor |
| 18 | paid | Extra paid classes in the subject | Binary | predictor |
| 19 | activities | Extra-curricular activities | Binary | predictor |
| 20 | nursery | Attended nursery school | Binary | predictor |
| 21 | higher | Wants higher education | Binary | predictor |
| 22 | internet | Internet at home | Binary | predictor |
| 23 | romantic | In a romantic relationship | Binary | predictor |
| 24 | famrel | Family relationship quality 1–5 | Ordinal | predictor |
| 25 | freetime | Free time after school 1–5 | Ordinal | predictor |
| 26 | goout | Going out with friends 1–5 | Ordinal | predictor |
| 27 | Dalc | Workday alcohol consumption 1–5 | Ordinal | predictor (sensitive) |
| 28 | Walc | Weekend alcohol consumption 1–5 | Ordinal | predictor (sensitive) |
| 29 | health | Current health 1 (very bad) – 5 (very good) | Ordinal | predictor |
| 30 | absences | School absences, 0 to 93 (per UCI) | Integer | key predictor |
| 31 | G1 | First-period grade, 0–20 | Integer | **Model B only** |
| 32 | G2 | Second-period grade, 0–20 | Integer | **Model B only** |
| 33 | G3 | Final grade, 0–20 (UCI: "output target") | Integer | **target** |

Grades relate to the course subject (Math or Portuguese). Observed ranges for each file are **[VERIFY]** after loading.

## 4. Target and G1/G2/G3

- **G3** is the final-year grade (issued in the 3rd period). It is the target for regression and the source of the Low/Medium/High category.
- **G1** and **G2** are the 1st and 2nd period grades. UCI states that G3 has a strong correlation with G1 and G2 and that predicting G3 without them is harder but "much more useful".
- EduLens therefore builds **Model A** (no G1/G2) and **Model B** (with G1/G2), and reports the difference rather than hiding it. The *measured* correlation values are computed in the app, not written here.

## 5. Performance Categories

Categories are derived from G3. Thresholds are configurable. Candidate methods (decision pending in Phase 2):

1. **Fixed bands on the 0–20 scale** (for example, a pass boundary and a "high" boundary). Requires a written justification; the pass mark and any grade-band convention should be cited from a source, not assumed.
2. **Data-driven quantiles** (for example, terciles of G3). Balanced classes, but thresholds depend on the sample.

No default is final until the choice is documented in `STATISTICAL_METHODS.md`.

## 6. Data Quality Notes

- UCI lists no missing values; EduLens still runs checks and reports counts.
- Duplicate check: rows are not guaranteed unique identifiers; exact duplicate rows are reported, not silently dropped.
- G3 = 0 records: inspect count and pattern (**[VERIFY]**) before deciding on handling (keep, flag, or run sensitivity analysis).
- `absences` has a long right tail (range to 93 per UCI); inspect with IQR rule and report.
- The UCI page text shows encoding artifacts in dashes (e.g., "â€“"); this only affects the web text, but CSV encoding should be checked on load.
- Ordinal coded variables (1–5 scales) are not true interval data; treatment (ordinal vs numeric) is stated wherever used.

## 7. Ethical Considerations

- Contains data about minors' school performance, family background and alcohol use; handle as sensitive even though the dataset is public.
- Sensitive or potentially stigmatizing attributes (sex, address, parental data, alcohol, romantic status) are used only where analytically justified, and any model use includes caveats.
- Findings are about this population; they do not generalize to other schools, countries or years without evidence.
- Association is not causation; the dashboard must not present factors as causes.
- Not for individual high-stakes decisions.

## 8. Limitations

- Two Portuguese secondary schools; data from the period around the 2008 paper (collection dates **[VERIFY]**).
- Self-reported questionnaire variables (study time, health, alcohol) may be biased.
- Binned variables (study time, travel time) lose detail.
- Subject-specific grades; results for Math and Portuguese may differ.
- Potential student overlap between the two files **[VERIFY]**.
- Observational data; confounding is likely.

## 9. Acquisition Instructions (to be finalized in Phase 2)

1. Download the zip from the official link above.
2. Extract into `data/raw/` and record file checksums in `data/README.md`.
3. Do not edit raw files; cleaning produces files in `data/processed/`.
4. Alternative: the `ucimlrepo` package (`fetch_ucirepo(id=320)`) per UCI; **[VERIFY]** whether it returns both course files or a single table before relying on it.
