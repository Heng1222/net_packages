# UCD-CVAE Version 2.1

This independent experiment learns one common malicious evidence gate, fourteen MITRE ATT&CK tactic evidence gates, and a 16-dimensional residual latent from frozen ModernBERT payload embeddings. Training never consumes tactic labels; Step2 labels are used only for post-training diagnostics.

Gate values are **uncalibrated evidence scores**, not probabilities. The inference policy emits recommendations only and never modifies a firewall.

## Commands

```powershell
uv run python experiments\ucd_cvae_v2_1\run_experiment.py --config experiments\ucd_cvae_v2_1\configs\default.yaml --stage all

uv run python experiments\ucd_cvae_v2_1\run_inference.py --checkpoint outputs\ucd_cvae_v2_1\RUN\checkpoints\gate_only_full_orthogonal.pt --input payloads.csv --text-col clean_payload_list --output predictions.jsonl

uv run python -m unittest discover -s experiments\ucd_cvae_v2_1\tests -v
```

`all` deliberately rebuilds the experiment's own ModernBERT cache from the source CSV. Use the separate stages to reuse a valid cache or evaluate an existing `--run-dir`.

## Geometry variants

- `full_orthogonal`: uncentered-SVD common component removal, deterministic completion of the one rank necessarily lost by removing that singular direction, then symmetric Löwdin orthogonalization before and after the learned near-identity projector.
- `common_removal_only`: removes the common component and normalizes each tactic without forcing pairwise tactic orthogonality.

The residual projection uses a QR basis for the complete concept span. ModernBERT latency depends on token length; only the fixed 768-to-15 gate head has constant dimensional cost.
