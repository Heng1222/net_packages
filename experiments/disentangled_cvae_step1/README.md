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

`Normal (TA9000)` is not a condition. `Sess_Tactic_predict` is kept only in prepared metadata for traceability. It is not used as a weak label or used for condition grouping. Test-set exported condition labels are derived from independent multi-label CVAE condition gates.

## Design

The experiment is isolated under `experiments/disentangled_cvae_step1/` and does not modify the previous `ae_cvae_tactic` experiment.

The pipeline has two stages:

1. `prepare`: stream the large Step1 CSV, parse labels as metadata, embed payload text in batches, and write reusable cache files.
2. `train`: load cached embeddings, build time splits, train a disentangled CVAE, and write metrics/reports.

The cached prepared dataset contains:

- `x.npy`: payload embeddings, shape `[N, 768]`
- `metadata.csv`: sample id, metadata label, time, payload hash, ISP/protocol metadata
- `manifest.json`: source/config fingerprint used to decide whether cache can be reused

The raw condition embeddings are 768-dimensional vectors produced by the same `nomic-ai/modernbert-embed-base` model. They are real CVAE condition input in this version. By default, each condition is still one tactic, but its text is now built from tactic-level keywords plus the complete top-level MITRE Enterprise ATT&CK v11.3 technique names under that tactic. Technique IDs and sub-techniques are omitted from the condition text. This avoids embedding the full prose descriptions with many shared filler words and should make the raw condition cosine similarity diagnostic more informative.

Because tactic descriptions still share a large common MITRE/security background, the model-used condition matrix applies a configurable geometry transform after raw embedding. The default subtracts the condition centroid, skips principal-direction removal, then row-normalizes the result. This is intended to reduce shared background semantics such as "adversary", "technique", and "network" while keeping condition-specific residual directions. For every sample, the encoder receives the payload embedding plus this full non-Normal condition matrix. The encoder learns `H` and one gate per condition. The decoder receives `H` plus the gated condition embeddings and reconstructs `x`.

### Condition Geometry Transform

The condition transform is configured under `conditions.geometry`:

```yaml
geometry:
  method: "common_component_removal"
  center: true
  remove_top_components: 0
  normalize: true
  strength: 1.0
```

This is not a contrastive loss and does not fine-tune the embedding model. It is a deterministic post-processing step applied only to the fixed condition matrix. The raw condition vectors are still saved for diagnostics, while the transformed vectors are the ones used by the encoder, decoder, gate decorrelation loss, and gated semantic summaries.

With `remove_top_components: 0`, the transform subtracts the condition centroid and then row-normalizes each condition vector. Setting `remove_top_components` to `1` additionally removes the first shared principal direction before row-normalization.

For the current 13 default ModernBERT condition vectors, this expands the condition space as follows:

| off-diagonal cosine summary | raw condition vectors | model-used condition vectors |
| --- | ---: | ---: |
| mean | 0.7545 | -0.0830 |
| median | 0.7519 | -0.0875 |
| max | 0.8414 | 0.2850 |

The lower model-used off-diagonal values mean the conditions are no longer clustered around the same shared MITRE/security background direction. A slightly negative average is expected after centering a small set of vectors; it means the condition directions are spread around the origin, not that the tactics are semantically opposite. If the transformed max cosine is still too high, increase `remove_top_components` to `1` or `2`. If the transform becomes too aggressive, lower `strength` to `0.5` or `0.75`.

Token length check on `clean_payload_list` with the ModernBERT tokenizer showed that almost all Step1 rows fit in 8192 tokens: median 94, p99 332, p99.9 650. Only 7 of 409,699 rows exceeded 8192 tokens, with the maximum at 45,700. Therefore the pipeline does not chunk ordinary rows. It chunks only overflow rows and averages their chunk embeddings so full-data preparation does not fail or silently truncate those rare long payloads.

```text
Payload text
  -> ModernBERT payload embedding x [768]
Condition keywords + technique names
  -> ModernBERT raw condition embeddings
  -> common-component removal
  -> model-used condition embeddings C_all [num_conditions, 768]

Encoder input:
  concat(x, flatten(C_all))

Encoder outputs:
  residual H [64]
  independent sigmoid gates [num_conditions]

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

Condition cosine similarity is reported as raw and model-used diagnostic heatmaps. The raw heatmap shows whether the embedding model collapses tactic descriptions into a shared background direction. The model-used heatmap shows the geometry actually fed to the CVAE after common-component removal.

`L_sparse`: sparse gate activation. This implements the meeting requirement "do not split unless needed." It encourages each sample to use fewer condition concepts, making the explanation simpler.

`L_gate_entropy`: gate binarization. Because condition use is treated as multi-label rather than a single softmax class, each gate is an independent sigmoid probability. The entropy term encourages gates to move toward clear off/on decisions instead of staying near 0.5.

`L_utility`: condition utility through ablation. Each gated condition is removed and the decoder is asked to reconstruct again. If removing a condition does not hurt reconstruction, the condition gate is probably decorative. The margin loss encourages active condition gates to carry useful information.

`L_residual_constraint`: residual-only limitation. The decoder is run with all condition gates removed. If `H` alone reconstructs almost as well as `H + gated C`, then the condition input is not doing the work. This loss encourages useful behavior information to move into the gated condition pathway.

Default weights:

```text
reconstruction: 1.0
kl: 1.0
decorrelation: 0.1
sparse: 1.0
gate_entropy: 0.01
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
- `logs/experiment.log`: stage progress plus per-epoch train/validation loss
- `checkpoints/disentangled_cvae.pt`
- `metrics/training_history.csv`
- `metrics/loss_summary.json`
- `metrics/condition_gate_summary.csv`
- `metrics/condition_ablation_delta_mse_summary.csv`
- `metrics/condition_raw_cosine_similarity.csv`
- `metrics/condition_cosine_similarity.csv`
- `metrics/leakage_report.json`
- `metrics/testset_condition_predictions.csv`
- `metrics/testset_subset_100.csv`
- `plots/condition_raw_cosine_similarity.png`
- `plots/condition_cosine_similarity.png`
- `plots/training_reconstruction_losses.png`
- `plots/umap_original_space.png`
- `plots/umap_h_space.png`
- `plots/umap_gated_c_space.png`
- `reports/report.md`

`Sess_Tactic_predict` is metadata only in this version. It is not used as a weak label, no tactic classifier/probe is trained, and no condition table is grouped by it. Test-set CSV exports `predicted_conditions` for multi-label gate activations; UMAP/PCA plots are colored by the model-derived highest active `predicted_condition` label.
