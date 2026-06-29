from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ORACLE_WARNING = (
    "CVAE oracle-condition classification is not a pure payload-only tactic prediction test. "
    "It evaluates whether latent representation becomes more tactic-aligned when conditioned "
    "on known tactic descriptions."
)


def _metric_table(title: str, value: dict[str, Any] | None) -> list[str]:
    lines = [f"## {title}", ""]
    if not value:
        return lines + ["Not run or unavailable.", ""]
    classification = value.get("classification", {})
    clustering = value.get("clustering", {})
    losses = value.get("losses", {}).get("test", {})
    reconstruction = value.get("reconstruction", {})
    flattened = {
        **({"recon_loss": reconstruction.get("test")} if reconstruction else {}),
        **losses,
        **classification,
        **clustering,
    }
    loss_keys = (
        ("recon_mse", "recon_nll", "kl_loss", "elbo", "negative_elbo")
        if "negative_elbo" in flattened
        else ("recon_loss", "kl_loss", "total_loss")
    )
    for key in ("accuracy", "macro_f1", "weighted_f1", *loss_keys, "silhouette", "nmi", "ari"):
        item = flattened.get(key)
        if item is not None:
            lines.append(f"- {key}: {item:.6f}" if isinstance(item, float) else f"- {key}: {item}")
    lines.append("")
    return lines


def write_report(
    path: str | Path,
    config: dict[str, Any],
    data_summary: dict[str, Any],
    results: dict[str, Any],
    warnings: list[str],
) -> None:
    lines = [
        "# AE / CVAE Tactic Latent Space Report",
        "",
        "## Experiment configuration",
        "",
        f"- Seed: {config.get('seed')}",
        f"- Input: {config.get('data', {}).get('input_path')}",
        f"- Normalization: {config.get('preprocessing', {}).get('normalization')}",
        f"- Payload embedder: {config.get('data', {}).get('embedder', {}).get('model_name')}",
        f"- Condition mode: {config.get('conditions', {}).get('condition_mode')}",
        "",
        "## Data and split summary",
        "",
        f"- Samples: {data_summary.get('num_samples')}",
        f"- Input dimension: {data_summary.get('input_dim')}",
        f"- Split counts: {data_summary.get('split_counts')}",
        f"- Label counts: {data_summary.get('label_counts')}",
        "",
    ]
    lines += _metric_table("AE results (10-class when labels are available)", results.get("ae"))
    lines += _metric_table("AE MITRE-only comparison", results.get("ae_mitre_only"))
    lines += _metric_table("CVAE oracle-condition results", results.get("cvae"))
    lines += _metric_table("Compatibility test", results.get("compatibility"))
    lines += ["## Condition ablation", ""]
    if results.get("ablation_summary"):
        summary_path = Path(results["ablation_summary"])
        lines.append(f"Summary CSV: `{summary_path}`")
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                columns = list(rows[0])
                lines += ["", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
                lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    else:
        lines.append("Not run or unavailable.")
    lines += [
        "",
        "## Leakage and interpretation warnings",
        "",
        f"> {ORACLE_WARNING}",
        "",
        "- AE latent classification tests whether the frozen payload embedding already contains tactic-level separability.",
        "- Compatibility testing pairs each payload with every candidate tactic description and is closer to tactic inference than oracle-conditioned classification.",
        "- Similar full/short/keywords/random/wrong results suggest that the CVAE may not be using tactic semantics.",
        "- Payload models operate on pretrained text embeddings rather than raw packets; results are limited by the embedding model's security-domain understanding.",
        "- Normal (TA9000) is excluded from CVAE because it is not a MITRE ATT&CK tactic; AE reports both ten-class and MITRE-only results.",
        "- Row-wise splitting allows identical payload text to occur in different splits and may inflate evaluation metrics.",
        "",
        "## Run warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    lines += [
        "",
        "## Interpretation notes",
        "",
        "Compare AE and CVAE only on the shared MITRE-only subset. Treat the two-sample Defense Evasion class as anecdotal rather than statistically reliable.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
