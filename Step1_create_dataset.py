from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd


FOLDER = "./Year=2022"
RAWDATA_FILENAME = "Step1_rawdata.csv"
CLEANED_FILENAME = "Step1_rawdata_cleaned.csv"
SUPPORTED_PICKLE_SUFFIXES = (".pickle", ".pickle.zst")
DEDUP_COLUMN = "clean_payload_list"
EMPTY_THRESHOLD = 0.5
PROGRESS_INTERVAL = 50_000


def is_supported_pickle_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in SUPPORTED_PICKLE_SUFFIXES)


def pickle_identity(path: Path) -> Path:
    name = path.name
    lower_name = name.lower()

    if lower_name.endswith(".pickle.zst"):
        return path.with_name(name[: -len(".pickle.zst")])
    if lower_name.endswith(".pickle"):
        return path.with_name(name[: -len(".pickle")])

    return path


def get_pickle_files(folder: str | Path) -> tuple[Path, list[Path]]:
    input_path = Path(folder).expanduser().resolve()

    if input_path.is_file():
        if not is_supported_pickle_file(input_path):
            raise ValueError(f"FOLDER points to a file, but it is not a supported pickle file: {input_path}")
        return input_path.parent, [input_path]

    if input_path.is_dir():
        raw_pickle_files = sorted(input_path.rglob("*.pickle"))
        zst_pickle_files = sorted(input_path.rglob("*.pickle.zst"))

        pickle_by_identity = {pickle_identity(path): path for path in zst_pickle_files}
        pickle_by_identity.update({pickle_identity(path): path for path in raw_pickle_files})
        pickle_files = sorted(pickle_by_identity.values())

        if not pickle_files:
            suffixes = ", ".join(SUPPORTED_PICKLE_SUFFIXES)
            raise FileNotFoundError(f"No supported pickle files ({suffixes}) found under: {input_path}")
        return input_path, pickle_files

    raise FileNotFoundError(f"FOLDER does not exist: {input_path}")


def read_pickle_as_dataframe(path: Path) -> pd.DataFrame:
    data = pd.read_pickle(path, compression="infer")

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()

    try:
        return pd.DataFrame(data)
    except Exception as exc:
        raise TypeError(f"Cannot convert pickle content to DataFrame: {path}") from exc


def resolve_output_path(output_file: str | Path | None, root_dir: Path, default_filename: str) -> Path:
    if output_file is None:
        return root_dir / default_filename

    return Path(output_file).expanduser().resolve()


def default_rawdata_path(folder: str | Path) -> Path:
    input_path = Path(folder).expanduser().resolve()
    root_dir = input_path.parent if input_path.is_file() else input_path
    return root_dir / RAWDATA_FILENAME


def create_rawdata(folder: str | Path, output_file: str | Path | None = None) -> Path:
    root_dir, pickle_files = get_pickle_files(folder)
    output_path = resolve_output_path(output_file, root_dir, RAWDATA_FILENAME)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if temp_output_path.exists():
        temp_output_path.unlink()

    expected_columns = None
    for index, path in enumerate(pickle_files, start=1):
        dataframe = read_pickle_as_dataframe(path)

        if expected_columns is None:
            expected_columns = list(dataframe.columns)
        elif list(dataframe.columns) != expected_columns:
            raise ValueError(f"Column mismatch in pickle file: {path}")

        dataframe.to_csv(
            temp_output_path,
            mode="w" if index == 1 else "a",
            header=index == 1,
            index=False,
            encoding="utf-8-sig" if index == 1 else "utf-8",
        )
        print(f"[{index}/{len(pickle_files)}] appended: {path}")

    temp_output_path.replace(output_path)

    return output_path


def set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def find_columns_to_drop(
    csv_path: Path,
    dedup_column: str,
    empty_threshold: float,
) -> tuple[list[str], list[int], int, list[float], int]:
    total_rows = 0
    empty_counts: list[int] | None = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.reader(input_handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc

        if dedup_column not in columns:
            raise KeyError(f"Column not found for deduplication: {dedup_column}")

        dedup_index = columns.index(dedup_column)
        column_count = len(columns)
        empty_counts = [0] * column_count

        for row_number, row in enumerate(reader, start=2):
            if len(row) != column_count:
                raise ValueError(
                    f"Unexpected column count at CSV record {row_number}: "
                    f"expected {column_count}, got {len(row)}"
                )

            total_rows += 1
            for index, value in enumerate(row):
                if value == "":
                    empty_counts[index] += 1

            if total_rows == 1 or total_rows % PROGRESS_INTERVAL == 0:
                print(f"[pass 1] scanned rows={total_rows}")

    if total_rows == 0 or empty_counts is None:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    empty_ratios = [count / total_rows for count in empty_counts]
    drop_indices = [index for index, ratio in enumerate(empty_ratios) if ratio > empty_threshold]

    return columns, drop_indices, total_rows, empty_ratios, dedup_index


def stable_payload_key(value: str) -> tuple[int, bytes]:
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).digest()
    return len(value), digest


def clean_rawdata(
    input_file: str | Path,
    output_file: str | Path,
    dedup_column: str = DEDUP_COLUMN,
    empty_threshold: float = EMPTY_THRESHOLD,
) -> Path:
    input_path = Path(input_file).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not 0 <= empty_threshold <= 1:
        raise ValueError("empty_threshold must be between 0 and 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_output_path.exists():
        temp_output_path.unlink()

    set_max_csv_field_size()
    columns, drop_indices, total_rows, empty_ratios, dedup_index = find_columns_to_drop(
        input_path,
        dedup_column=dedup_column,
        empty_threshold=empty_threshold,
    )
    drop_index_set = set(drop_indices)
    keep_indices = [index for index in range(len(columns)) if index not in drop_index_set]
    keep_columns = [columns[index] for index in keep_indices]

    print(f"[pass 1] rows={total_rows}, columns={len(columns)}")
    print(f"[pass 1] dropping columns={len(drop_indices)}")
    for index in drop_indices:
        print(f"  - {columns[index]}: {empty_ratios[index]:.2%} empty")

    seen_payloads: set[tuple[int, bytes]] = set()
    written_rows = 0
    duplicate_rows = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        with temp_output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
            reader = csv.reader(input_handle)
            writer = csv.writer(output_handle, lineterminator="\n")

            columns = next(reader)
            column_count = len(columns)
            writer.writerow(keep_columns)

            for processed_rows, row in enumerate(reader, start=1):
                if len(row) != column_count:
                    raise ValueError(
                        f"Unexpected column count at CSV record {processed_rows + 1}: "
                        f"expected {column_count}, got {len(row)}"
                    )

                payload_key = stable_payload_key(row[dedup_index])
                if payload_key in seen_payloads:
                    duplicate_rows += 1
                    should_write = False
                else:
                    seen_payloads.add(payload_key)
                    should_write = True

                if should_write:
                    writer.writerow([row[index] for index in keep_indices])
                    written_rows += 1

                if processed_rows == 1 or processed_rows % PROGRESS_INTERVAL == 0:
                    print(
                        f"[pass 2] processed rows={processed_rows}, "
                        f"written_rows={written_rows}, duplicate_rows={duplicate_rows}"
                    )

    temp_output_path.replace(output_path)
    print(f"Saved cleaned CSV: {output_path}")
    print(f"Original rows: {total_rows}")
    print(f"Output rows: {written_rows}")
    print(f"Duplicate rows removed by {dedup_column}: {duplicate_rows}")
    print(f"Columns removed with > {empty_threshold:.0%} empty values: {len(drop_indices)}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Step1 rawdata CSV and cleaned CSV.")
    parser.add_argument("--folder", default=FOLDER, help="Folder or pickle file used to create rawdata.")
    parser.add_argument("--raw-output", default=None, help="Raw CSV output path.")
    parser.add_argument("--clean-output", default=None, help="Cleaned CSV output path.")
    parser.add_argument("--dedup-column", default=DEDUP_COLUMN, help="Column used to remove duplicate rows.")
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=EMPTY_THRESHOLD,
        help="Drop columns whose empty ratio is greater than this value.",
    )
    parser.add_argument(
        "--skip-rawdata",
        action="store_true",
        help="Skip pickle merge and clean an existing rawdata CSV.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Only create rawdata CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_rawdata:
        rawdata_path = Path(args.raw_output).expanduser().resolve() if args.raw_output else default_rawdata_path(args.folder)
    else:
        rawdata_path = create_rawdata(args.folder, output_file=args.raw_output)
        print(f"Saved raw data: {rawdata_path}")

    if args.no_clean:
        return

    cleaned_output_path = (
        Path(args.clean_output).expanduser().resolve()
        if args.clean_output
        else rawdata_path.with_name(CLEANED_FILENAME)
    )
    clean_rawdata(
        input_file=rawdata_path,
        output_file=cleaned_output_path,
        dedup_column=args.dedup_column,
        empty_threshold=args.empty_threshold,
    )


if __name__ == "__main__":
    main()
