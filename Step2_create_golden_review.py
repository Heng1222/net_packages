from pathlib import Path

import pandas as pd


FILE = "./Year=2022/Step1_rawdata.csv"
OUTPUT_FILENAME = "Step2_golden_review.csv"
RANDOM_SEED = 42
SAMPLE_PER_CLASS = 200
TARGET_COLUMN = "Sess_Tactic_predict"
COLUMN_KEEP_LIST = [
    "Session_ID",
    "Datetime",
    "Protocol",
    "ip_src",
    "ip_dst",
    "payload_list",
    "clean_payload_list",
    "Sess_Tactic_predict",
    "Sess_Technique_predict",
    "Sess_SubTechnique_predict",
    "Sess_Malicious_Score",
    "Sess_Pkt_ImportantScore",
]


def filter_keep_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    if not COLUMN_KEEP_LIST:
        return dataframe

    missing_columns = [column for column in COLUMN_KEEP_LIST if column not in dataframe.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Columns not found for output: {missing_text}")

    return dataframe[COLUMN_KEEP_LIST]


def create_golden_review(file: str | Path) -> Path:
    csv_path = Path(file).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rawdata = pd.read_csv(csv_path)

    if TARGET_COLUMN not in rawdata.columns:
        raise KeyError(f"Column not found in {csv_path.name}: {TARGET_COLUMN}")

    valid_rawdata = rawdata.dropna(subset=[TARGET_COLUMN])
    class_counts = valid_rawdata[TARGET_COLUMN].value_counts(sort=False)
    if class_counts.empty:
        raise ValueError(f"No non-empty classes found in column: {TARGET_COLUMN}")

    eligible_classes = class_counts[class_counts >= SAMPLE_PER_CLASS]
    if eligible_classes.empty:
        raise ValueError(f"No classes have at least {SAMPLE_PER_CLASS} rows in column: {TARGET_COLUMN}")

    skipped_classes = class_counts[class_counts < SAMPLE_PER_CLASS]
    if not skipped_classes.empty:
        detail = ", ".join(f"{label}={count}" for label, count in skipped_classes.items())
        print(f"Skipped classes with fewer than {SAMPLE_PER_CLASS} rows: {detail}")

    valid_rawdata = valid_rawdata[valid_rawdata[TARGET_COLUMN].isin(eligible_classes.index)]

    golden_review = (
        valid_rawdata.groupby(TARGET_COLUMN, group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    golden_review = filter_keep_columns(golden_review)

    output_path = csv_path.parent / OUTPUT_FILENAME
    golden_review.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main() -> None:
    golden_review_path = create_golden_review(FILE)
    print(f"Saved golden review: {golden_review_path}")


if __name__ == "__main__":
    main()
