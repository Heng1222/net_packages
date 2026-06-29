# Local data directory

This directory is kept in git only as a data placement target. Actual training data, generated CSVs, embedding caches, and experiment outputs are ignored.

Place raw source files anywhere under this directory:

```text
Year=2022/Month=10/Day=05/tagging_table_20221005_00_cht_http.pickle.zst
```

Expected generated or curated files for the default workflow:

```text
Step1_rawdata.csv
Step1_rawdata_cleaned.csv
Step2_golden_review.csv
Step2_golden_review_with_Tactic.csv
```

`Step2_golden_review_with_Tactic.csv` is required by the AE/CVAE tactic configs and must include a `Tactic` column.
