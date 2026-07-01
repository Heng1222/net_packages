# Step1 Disentangled CVAE Experiment

## Purpose

This experiment implements the 2026-06-23 meeting direction: the goal is not only to classify payloads or inspect one latent `H` space. The goal is to learn a behavior representation that extracts condition-related concepts from the input through an explicit CVAE condition pathway.

The first version uses the cleaned Step1 packet/session dataset:

- Input CSV: `Year=2022/Step1_rawdata_cleaned.csv`
- Default payload column: `clean_payload_list`
- Metadata label column: `Sess_Tactic_predict`
- Embedding model: `nomic-ai/modernbert-embed-base`
- Primary split: time split, 70% train, 15% validation, 15% test
- Payload overflow policy: `chunk_mean`

`Normal (TA9000)` is not a condition. `Sess_Tactic_predict` is kept only in prepared metadata for traceability. It is not used as a weak label or used for condition grouping. Test-set plot colors and exported prediction labels are derived from softmax-normalized CVAE condition gates.

## Design

The experiment is isolated under `experiments/disentangled_cvae_step1/` and does not modify the previous `ae_cvae_tactic` experiment.

The pipeline has two stages:

1. `prepare`: stream the large Step1 CSV, parse labels as metadata, embed payload text in batches, and write reusable cache files.
2. `train`: load cached embeddings, build time splits, train a disentangled CVAE, and write metrics/reports.

The cached prepared dataset contains:

- `x.npy`: payload embeddings, shape `[N, 768]`
- `metadata.csv`: sample id, metadata label, time, payload hash, ISP/protocol metadata
- `manifest.json`: source/config fingerprint used to decide whether cache can be reused

The condition embeddings are 768-dimensional vectors produced by the same `nomic-ai/modernbert-embed-base` model. They are real CVAE condition input in this version. For every sample, the encoder receives the payload embedding plus the full non-Normal condition matrix. The encoder learns `H` and one gate per condition. The decoder receives `H` plus the gated condition embeddings and reconstructs `x`.

Token length check on `clean_payload_list` with the ModernBERT tokenizer showed that almost all Step1 rows fit in 8192 tokens: median 94, p99 332, p99.9 650. Only 7 of 409,699 rows exceeded 8192 tokens, with the maximum at 45,700. Therefore the pipeline does not chunk ordinary rows. It chunks only overflow rows and averages their chunk embeddings so full-data preparation does not fail or silently truncate those rare long payloads.

```text
Payload text
  -> ModernBERT payload embedding x [768]
Condition descriptions
  -> ModernBERT condition embeddings C_all [num_conditions, 768]

Encoder input:
  concat(x, flatten(C_all))

Encoder outputs:
  residual H [64]
  gates [num_conditions]

Decoder input:
  concat(H, flatten(gates * C_all))

Decoder output:
  reconstructed x [768]
```

## Loss Design

The loss is designed to prevent the model from hiding everything in one latent vector while still preserving reconstruction quality.

`L_rec`: reconstruction negative log-likelihood. This keeps the representation faithful to the input embedding. Without it, gates could become easy to separate but unrelated to the original behavior.

`L_kl`: KL on residual `H`. `H` should be a compact residual channel, not an unlimited memory bank. The KL term keeps it close to a normal prior and reduces the chance that all semantic information bypasses the condition pathway.

`L_decorrelation`: condition-gate independence. It penalizes simultaneously activating condition vectors that are highly similar. This addresses the meeting concern that similar C spaces can split the same information arbitrarily and become hard to interpret.

Condition cosine similarity is reported as a diagnostic heatmap only. It is not optimized by the model in this version.

`L_sparse`: sparse gate activation. This implements the meeting requirement "do not split unless needed." It encourages each sample to use fewer condition concepts, making the explanation simpler.

`L_utility`: condition utility through ablation. Each gated condition is removed and the decoder is asked to reconstruct again. If removing a condition does not hurt reconstruction, the condition gate is probably decorative. The margin loss encourages active condition gates to carry useful information.

`L_residual_constraint`: residual-only limitation. The decoder is run with all condition gates removed. If `H` alone reconstructs almost as well as `H + gated C`, then the condition input is not doing the work. This loss encourages useful behavior information to move into the gated condition pathway.

Default weights:

```text
reconstruction: 1.0
kl: 1.0
decorrelation: 0.1
sparse: 0.001
utility: 0.5
residual_constraint: 0.5
```

## Commands

Prepare only:

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage prepare
```

Train only, reusing prepared data:

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage train
```

Run both stages:

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage all
```

Run tests:

```powershell
uv run python -m unittest discover -s experiments\disentangled_cvae_step1\tests -v
```

## Outputs

Each run writes to `outputs/disentangled_cvae_step1/<timestamp>/`:

- `config_resolved.yaml`
- `environment.json`
- `logs/`
- `checkpoints/disentangled_cvae.pt`
- `metrics/training_history.csv`
- `metrics/loss_summary.json`
- `metrics/condition_gate_summary.csv`
- `metrics/condition_ablation_delta_mse_summary.csv`
- `metrics/condition_cosine_similarity.csv`
- `metrics/leakage_report.json`
- `metrics/testset_condition_predictions.csv`
- `metrics/testset_subset_100.csv`
- `plots/condition_cosine_similarity.png`
- `plots/training_reconstruction_losses.png`
- `plots/umap_original_space.png`
- `plots/umap_h_space.png`
- `plots/umap_gated_c_space.png`
- `reports/report.md`

`Sess_Tactic_predict` is metadata only in this version. It is not used as a weak label, no tactic classifier/probe is trained, and no condition table is grouped by it. Test-set UMAP/PCA plots are colored by the model-derived `predicted_condition` label.
