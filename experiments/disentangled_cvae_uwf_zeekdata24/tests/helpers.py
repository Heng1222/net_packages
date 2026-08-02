from __future__ import annotations

from typing import Any

import pandas as pd

from experiments.disentangled_cvae_uwf_zeekdata24.download import REQUIRED_COLUMNS


def uwf_row(uid: str, tactic: str = "none", technique: str = "none", index: int = 0) -> dict[str, Any]:
    row = {column: "none" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "community_id": f"community-{uid}",
            "conn_state": "SF",
            "duration": str(0.1 + index),
            "history": "ShADadFf",
            "src_ip_zeek": "143.88.1.18",
            "src_port_zeek": str(50000 + index),
            "dest_ip_zeek": "143.88.2.20",
            "dest_port_zeek": "445",
            "local_orig": "false",
            "local_resp": "false",
            "missed_bytes": "0",
            "orig_bytes": str(100 + index),
            "orig_ip_bytes": str(200 + index),
            "orig_pkts": "4",
            "proto": "tcp",
            "resp_bytes": str(300 + index),
            "resp_ip_bytes": str(400 + index),
            "resp_pkts": "5",
            "service": "smb,ntlm",
            "ts": str(1_700_000_000 + index),
            "uid": uid,
            "datetime": f"2024-03-{1 + index % 20:02d}T00:00:00Z",
            "label_tactic": tactic,
            "label_technique": technique,
            "label_binary": "False" if technique == "none" else "True",
            "label_cve": "none",
        }
    )
    return row


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(index=False).encode("utf-8")

