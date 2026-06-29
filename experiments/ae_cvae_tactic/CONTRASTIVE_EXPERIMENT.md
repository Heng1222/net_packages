# Payload-to-description contrastive CVAE experiment

This experiment is intentionally separate from `run_experiment.py`.
It reuses the existing data loading, deterministic split, preprocessing, condition loading,
and evaluation utilities, but uses a new model, trainer, checkpoint name, report, and output root.
The original AE/CVAE workflow is not invoked or overwritten.

## Objective

For payload embedding `x`, its oracle condition `c`, and all tactic condition embeddings `C`:

```text
negative_ELBO = Gaussian_NLL(x, x_recon) + KL(q(z | x, c) || N(0, I))
logits        = normalize(payload_projector(x)) @ normalize(C).T / temperature
contrastive   = cross_entropy(logits, true_tactic_index)
total_loss    = negative_ELBO + contrastive_weight * contrastive
```

The contrastive payload branch receives `x` only. It cannot inspect the oracle `c` used by the
CVAE branch. Condition embeddings use a frozen identity projection so the candidate geometry
remains the pretrained text embedding geometry.

Each batch is compared against the complete unique candidate matrix rather than using batch
rows as negatives. Payloads belonging to the same tactic therefore are not false negatives.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\ae_cvae_tactic\run_contrastive_experiment.py `
  --config experiments\ae_cvae_tactic\configs\contrastive.yaml
```

Outputs are written under `outputs/ae_cvae_tactic_contrastive/<timestamp>/` and include:

- `checkpoints/contrastive_cvae.pt`
- `logs/contrastive_cvae_history.csv`
- `metrics/contrastive_cvae_metrics.json`
- `metrics/contrastive_score_matrix.npz`
- `metrics/contrastive_retrieval_results.csv`
- `latent/contrastive_payload_*_latent.npz`
- `latent/contrastive_cvae_*_latent.npz`
- `reports/contrastive_report.md`

## Interpretation limit

This first experiment aligns payloads to nine fixed semantic prototypes. It can still learn a
closed-set nine-class mapping. Evidence of broader semantic use requires a follow-up evaluation
with held-out paraphrases or descriptions not used during training.
