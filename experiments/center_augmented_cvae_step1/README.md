# Center-Augmented CVAE Step1

This experiment trains a fully unsupervised CVAE on L2-normalized ModernBERT embeddings from
`Year=2022/Step1_rawdata_cleaned.csv`. It is an independent package and does not import code from
the other experiment packages.

## Representation hypothesis

Let the 13 MITRE tactic text embeddings be `p_i`:

```text
centroid = mean(p_i)
centered_i = p_i - centroid
p_i = centroid + centered_i
```

The model treats the centroid as a fourteenth common condition. Fixed semantic gates are computed
directly from payload/condition cosine similarity; there is no learned gate projector:

```text
g_i = sigmoid(cosine(x, normalize(condition_i)) / temperature)
```

The encoder receives `x` plus all fourteen fixed 768-dimensional decode conditions. The decoder is
explicitly additive:

```text
condition_part = g_common * centroid + sum(g_i * centered_i)
residual_part = ResidualDecoder(z)
x_recon = residual_part + condition_part
```

Training uses only reconstruction MSE and KL. `Sess_Tactic_predict` is retained in prepared metadata
for descriptive reports but is never loaded by the model, dataloader, or loss. Step2 golden labels are
read only after training for a transductive semantic-alignment diagnostic.

## Commands

Run tests:

```powershell
uv run python -m unittest discover `
  -s experiments\center_augmented_cvae_step1\tests -v
```

Prepare the full payload embedding cache:

```powershell
uv run python experiments\center_augmented_cvae_step1\run_experiment.py `
  --config experiments\center_augmented_cvae_step1\configs\default.yaml `
  --stage prepare
```

Train and evaluate after preparation:

```powershell
uv run python experiments\center_augmented_cvae_step1\run_experiment.py `
  --config experiments\center_augmented_cvae_step1\configs\default.yaml `
  --stage train
```

Run the complete pipeline:

```powershell
uv run python experiments\center_augmented_cvae_step1\run_experiment.py `
  --config experiments\center_augmented_cvae_step1\configs\default.yaml `
  --stage all
```

The first ModernBERT run may need to download model weights. Tests use an offline hashing embedder.

## Outputs

Prepared data is stored at:

```text
outputs/center_augmented_cvae_step1/prepared/step1_clean_payload_modernbert/
```

Timestamped runs contain:

```text
outputs/center_augmented_cvae_step1/YYYY-MM-DD_HHMMSS/
  checkpoints/   main, plain-VAE, and random-condition checkpoints
  embeddings/    raw, centroid, centered, decode, and gate matrices
  metrics/       histories, leakage, gates, ablations, predictions, golden diagnostic
  plots/         condition geometry and training curves
  reports/       concise Markdown report
```

## Required comparisons

The runner evaluates the main model against a same-capacity plain VAE and a random-condition CVAE.
For the main and random-condition models it measures full reconstruction, residual-only, common-only,
zero-common, shuffled gates, and every tactic-condition ablation.

Good reconstruction alone is not evidence of semantic disentanglement. If random conditions perform
the same as MITRE conditions, shuffled gates do not hurt reconstruction, or residual-only reconstruction
matches the full model, the semantic decomposition hypothesis is not supported.
