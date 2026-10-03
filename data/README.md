# Raw data files

## Official source

UCI Student Performance dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
- Download archive: https://archive.ics.uci.edu/static/public/320/student+performance.zip

For a fresh setup, download that archive and extract the two course CSVs into this directory.
The CSVs and reference files listed below were already supplied in the current workspace; they
were inspected in place and were not downloaded again or modified. The archive member listing was
not independently checked because the archive was not present locally.

## Supplied files and verified format

| File | Description | Observed format |
|---|---|---|
| `student-por.csv` | Portuguese course data; primary EduLens source | ASCII-only (strict UTF-8 decode succeeds), semicolon-delimited, 649 rows × 33 columns |
| `student-mat.csv` | Mathematics course data; optional, loaded separately | ASCII-only (strict UTF-8 decode succeeds), semicolon-delimited, 395 rows × 33 columns |
| `student.txt` | Supplied variable descriptions | UTF-8 text |
| `student-merge.R` | Supplied course-matching example | UTF-8 text |

Because every byte in the CSVs is ASCII, their original encoding cannot be uniquely distinguished
from the file contents alone; the loader decodes them as UTF-8.

The data loader never joins the course datasets. It accepts uploaded CSV bytes in memory and does
not write uploaded content to disk. Uploaded CSVs must contain at least 20 data rows by default.
To explicitly regenerate the cleaned Portuguese export, run this command from the repository root:

```powershell
python -m src.export_cleaned
```

The command uses the Phase 2 loader and cleaning pipeline and overwrites
`data/processed/student-por-cleaned.csv`, creating `data/processed/` if needed. The output includes
the derived `G3_zero_flag` column. The raw files under `data/raw/` are read-only inputs to the
export and are never rewritten. Processed outputs are git-ignored and can be regenerated at any
time. For measured data-quality and class-count details, see
[`../DATASET.md`](../DATASET.md).

## SHA-256 checksums

Checksums below were computed from the exact local file bytes on 2026-10-03.

| File | SHA-256 |
|---|---|
| `student-por.csv` | `a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f` |
| `student-mat.csv` | `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80` |
| `student.txt` | `f8d3e734e237071312790ca667330c66d75e706317f8d5e2125c479d70e962c1` |
| `student-merge.R` | `a04f4ef19551c319428f642269c125faa21c521045946beead1c32e656269d31` |
