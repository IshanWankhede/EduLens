# EduLens: Dataset

Official repository facts below are attributed to the UCI page and the supplied `student.txt`.
Measured file facts below were computed from the local raw files during Phase 2. The supplied
files were not downloaded again.

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

The local `data/raw/` copy contains `student-por.csv`, `student-mat.csv`, `student.txt`, and
`student-merge.R` (plus the repository placeholder). The extracted files were inspected directly.
Both CSVs contain only ASCII bytes and decode strictly as UTF-8; since ASCII is also valid in
several encodings, the original encoding cannot be uniquely identified from these bytes alone.
Both are semicolon-delimited. The Portuguese file has 649 rows and 33 columns, matching UCI's
listed instance count; the Mathematics file has 395 rows and 33 columns.

The exact member listing of the official download archive was not independently verified: no
archive is present locally, and Phase 2 explicitly did not download it again. The
local extracted-file inventory is recorded in `data/README.md`. The acquisition link remains the
official UCI download.

EduLens v1 uses `student-por.csv` as the primary file; `student-mat.csv` is optional and is loaded
separately. Following the keys in `student-merge.R`, a join on shared profile attributes yields
382 matched join rows from 366 shared key groups. Of those groups, 358 are one-to-one across files
and 8 have repeated key values in at least one file. The key identifies matching attribute
profiles, not a unique person identifier; therefore, the join count cannot prove 382 distinct
students. No combined dataset is created.

### Phase 2 file checks

Encoding was tested by strict decoding, and the delimiter was detected from the CSV header.
Missing values count cells; exact duplicates count rows after the first identical row. Performance
categories use the project-approved fixed G3 bands: Low `< 10`, Medium `10–13` inclusive, and High
`>= 14`.

| Course file | Rows × columns | G3 range | G3 = 0 | Absences range | Missing cells | Exact duplicate rows | Low | Medium | High |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `student-por.csv` | 649 × 33 | 0–19 | 15 | 0–32 | 0 | 0 | 100 | 355 | 194 |
| `student-mat.csv` | 395 × 33 | 0–20 | 38 | 0–75 | 0 | 0 | 130 | 165 | 100 |

Rows with `G3 = 0` are retained unchanged. The cleaned in-memory frame adds `G3_zero_flag` so
these records can be identified without silently excluding them. Neither outliers nor duplicate
rows are removed by Phase 2.

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

Grades relate to the course subject (Math or Portuguese). The observed G3 ranges are 0–19 for
Portuguese and 0–20 for Mathematics, as shown in the Phase 2 file-check table.

`student.txt` contains all 33 variable names represented in this table. Its descriptions are
compatible with this table's condensed descriptions. The text calls the coded education, travel,
study-time, failure, family-relation, leisure, going-out, alcohol, and health measures "numeric";
EduLens classifies these ordered codes as ordinal. The file repeats the number `31` for G1 and G2,
then labels G3 as `32`; this table numbers them sequentially as 31, 32, and 33. No other variable
name or value-definition discrepancy was found in the supplied variable description.

## 4. Target and G1/G2/G3

- **G3** is the final-year grade (issued in the 3rd period). It is the target for regression and the source of the Low/Medium/High category.
- **G1** and **G2** are the 1st and 2nd period grades. UCI states that G3 has a strong correlation with G1 and G2 and that predicting G3 without them is harder but "much more useful".
- EduLens therefore builds **Model A** (no G1/G2) and **Model B** (with G1/G2), and reports the difference rather than hiding it. The *measured* correlation values are computed in the app, not written here.

## 5. Performance Categories

Categories are derived from G3 using the project-selected fixed bands:

**Low** `< 10`, **Medium** `10–13` inclusive, **High** `>= 14`.

These are EduLens analysis categories, not asserted as an official grading or pass convention. The
configured method and cutoffs are returned with derived categories; class counts are computed from
the loaded course file.

## 6. Data Quality Notes

- UCI lists no missing values; EduLens still runs checks and reports counts.
- Duplicate check: rows are not guaranteed unique identifiers; exact duplicate rows are reported, not silently dropped.
- G3 = 0 records: 15 in Portuguese and 38 in Mathematics. They are retained and flagged in the
  in-memory cleaned frame; no sensitivity-analysis behavior is added in this phase.
- Observed `absences` ranges are 0–32 in Portuguese and 0–75 in Mathematics (the broader UCI
  variable description allows values up to 93); inspect outliers with the IQR rule and report.
- The UCI page text shows encoding artifacts in dashes (e.g., "â€“"); this only affects the web text, but CSV encoding should be checked on load.
- Ordinal coded variables (1–5 scales) are not true interval data; treatment (ordinal vs numeric) is stated wherever used.

## 7. Ethical Considerations

- Contains data about minors' school performance, family background and alcohol use; handle as sensitive even though the dataset is public.
- Sensitive or potentially stigmatizing attributes (sex, address, parental data, alcohol, romantic status) are used only where analytically justified, and any model use includes caveats.
- Findings are about this population; they do not generalize to other schools, countries or years without evidence.
- Association is not causation; the dashboard must not present factors as causes.
- Not for individual high-stakes decisions.

## 8. Limitations

- Two Portuguese secondary schools; the supplied local files do not state collection dates. The
  associated 2008 paper is not sufficient evidence to assert exact collection dates.
- Self-reported questionnaire variables (study time, health, alcohol) may be biased.
- Binned variables (study time, travel time) lose detail.
- Subject-specific grades; results for Math and Portuguese may differ.
- The supplied R script matches files on 13 shared attributes; its inner join yields 382 rows.
  Because 8 shared key groups are non-unique, this is not conclusive identification of distinct
  students. The files remain separate.
- Observational data; confounding is likely.

## 9. Acquisition Instructions

1. For a fresh local setup, download the zip from the official link above and extract its course
   files into `data/raw/`. In the current workspace, the requested source files were already
   present and were not downloaded again.
2. See `data/README.md` for the supplied local-file inventory and SHA-256 checksums. The archive
   member list was not independently inspected because the archive was not downloaded.
3. Do not edit raw files; cleaning produces in-memory output in Phase 2, and processed files are
   only written by a later explicit workflow.
4. EduLens reads the individual course CSV files directly. The `ucimlrepo` retrieval behavior was
   not tested and is not relied on.
