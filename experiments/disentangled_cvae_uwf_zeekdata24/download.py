from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REQUIRED_COLUMNS = (
    "community_id",
    "conn_state",
    "duration",
    "history",
    "src_ip_zeek",
    "src_port_zeek",
    "dest_ip_zeek",
    "dest_port_zeek",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "orig_bytes",
    "orig_ip_bytes",
    "orig_pkts",
    "proto",
    "resp_bytes",
    "resp_ip_bytes",
    "resp_pkts",
    "service",
    "ts",
    "uid",
    "datetime",
    "label_tactic",
    "label_technique",
    "label_binary",
    "label_cve",
)

SOURCE_CATEGORIES = (
    "Benign",
    "Credential_Access",
    "Defense_Evasion",
    "Exfiltration",
    "Initial_Access",
    "Persistence",
    "Privilege_Escalation",
    "Reconnaissance",
)


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "network-payload-cvae-experiments/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _csv_href(index_html: bytes) -> str:
    text = index_html.decode("utf-8", errors="replace")
    matches = re.findall(r'href=["\']([^"\']*part-[^"\']+\.csv)["\']', text, flags=re.IGNORECASE)
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected exactly one part-*.csv link in UWF index; found {unique}")
    return unique[0]


def _validate_csv_bytes(content: bytes) -> None:
    header = content.splitlines()[0].decode("utf-8-sig", errors="strict") if content else ""
    columns = tuple(header.split(","))
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Downloaded UWF CSV is missing required columns: {missing}")


def download_dataset(
    data_config: dict[str, Any],
    force: bool = False,
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> Path:
    raw_dir = Path(data_config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(data_config["source_base_url"]).rstrip("/")
    records: list[dict[str, Any]] = []
    for category in map(str, data_config["source_categories"]):
        destination = raw_dir / f"{category}.csv"
        index_url = f"{base_url}/{category}/"
        index_html = fetch_bytes(index_url)
        href = _csv_href(index_html)
        file_url = href if href.startswith(("http://", "https://")) else f"{index_url}{href}"
        reused = destination.is_file() and not force
        if reused:
            content = destination.read_bytes()
        else:
            content = fetch_bytes(file_url)
            _validate_csv_bytes(content)
            temporary = destination.with_suffix(".csv.tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
        _validate_csv_bytes(content)
        records.append(
            {
                "category": category,
                "url": file_url,
                "path": str(destination),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "reused": reused,
            }
        )
    manifest_path = raw_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "UWF-ZeekData24",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "homepage": "https://datasets.uwf.edu/",
                "paper": "https://doi.org/10.3390/data10050059",
                "license": "CC BY 4.0",
                "required_columns": list(REQUIRED_COLUMNS),
                "files": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path
