from __future__ import annotations

import argparse
from pathlib import Path

from Step1_create_dataset import (
    CLEANED_FILENAME,
    DEDUP_COLUMN,
    EMPTY_THRESHOLD,
    RAWDATA_FILENAME,
    clean_rawdata,
)


INPUT_FILE = Path("./Year=2022") / RAWDATA_FILENAME
OUTPUT_FILE = Path("./Year=2022") / CLEANED_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Step1 rawdata CSV.")
    parser.add_argument("--input", default=str(INPUT_FILE), help="Input Step1 rawdata CSV.")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output cleaned CSV.")
    parser.add_argument("--dedup-column", default=DEDUP_COLUMN, help="Column used to remove duplicate rows.")
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=EMPTY_THRESHOLD,
        help="Drop columns whose empty ratio is greater than this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_rawdata(
        input_file=args.input,
        output_file=args.output,
        dedup_column=args.dedup_column,
        empty_threshold=args.empty_threshold,
    )


if __name__ == "__main__":
    main()
