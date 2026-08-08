from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ucd_cvae_v2_1.inference import GateInferenceEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCD-CVAE v2.1 gate-only inference")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--text-col", default="clean_payload_list")
    parser.add_argument("--id-col", default=None); parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); input_path = Path(args.input); output_path = Path(args.output)
    frame = pd.read_json(input_path, lines=True) if input_path.suffix.lower() == ".jsonl" else pd.read_csv(input_path)
    if args.text_col not in frame: raise ValueError(f"Missing text column: {args.text_col}")
    ids = frame[args.id_col].astype(str).tolist() if args.id_col else None
    engine = GateInferenceEngine.from_checkpoint(args.checkpoint, args.device)
    results = engine.predict_texts(frame[args.text_col].fillna("").astype(str).tolist(), ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        flat = []
        for item in results:
            row = {key: value for key, value in item.items() if key not in {"tactic_evidence", "top_tactics", "latency_ms"}}
            row["tactic_evidence"] = json.dumps(item["tactic_evidence"], ensure_ascii=False)
            row["top_tactics"] = json.dumps(item["top_tactics"], ensure_ascii=False)
            row["latency_ms"] = json.dumps(item["latency_ms"]); flat.append(row)
        pd.DataFrame(flat).to_csv(output_path, index=False)
    else:
        output_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__": main()
