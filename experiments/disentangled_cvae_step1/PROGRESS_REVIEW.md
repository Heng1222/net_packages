# Disentangled CVAE Step1 Progress Review

## Bottom line

The core hypothesis is coherent: represent a payload as shared/residual information
`H` plus independently gated tactic concepts, then use the gates as tactic evidence.
The implementation has the right high-level pieces (payload projector, tactic
prototypes, sigmoid gates, residual latent, gated decoder, ablation diagnostics).

It is not yet valid to interpret the real-data gate values as "how much of each
tactic". The present real-data run is incomplete and the current time-split test set
has no usable golden labels. A controlled synthetic recovery check does pass, so the
architecture has the capacity to recover known additive condition mixtures. The
next milestone is semantic validation on held-out real labels, before loss or model
capacity optimization.

## Verified current state

- Prepared manifest/metadata describe 409,699 rows with 768-dimensional embeddings,
  but the prepared directory currently has no `x.npy`; preparation is incomplete.
- The existing `2026-06-27_234255` run has split/leakage files and a condition cache,
  but no checkpoint, training history, predictions, or final report.
- Golden CSV: 2,000 labeled rows; 1,196 Session IDs match prepared metadata.
- Known-condition labels by current time split: train 1,142, validation 35, test 0.
- `Normal (TA9000)` is deliberately not a condition; 19 matched rows are Normal and
  are currently ignored by the semantic loss rather than trained as all-zero gates.
- Several tactics have no matched examples and some have only one or a few. Macro
  evaluation across all 13 tactics is not supportable with the current labels.

Every future run now writes `metrics/behavior_supervision_by_split.json` and warns
when its semantic test evaluation has no gold labels.

## Capability check

`concept_validation.py` creates held-out synthetic samples with known multi-tactic
mixtures and a separate shared residual. It trains the same gate/H model and uses
fixed pass criteria.

Seed 42 result:

- multi-label macro F1: 1.0000 (required >= 0.80)
- gate/target correlation: 0.9999 (required >= 0.75)
- full reconstruction MSE: 0.0272
- H-only reconstruction MSE: 1.3801
- condition reconstruction gain: 1.3529 (required >= 0.05)
- shuffled-target macro F1: 0.3922

This establishes implementation capacity under an identifiable additive data
generating process. It does not establish that ModernBERT payload embeddings and
MITRE tactic text share that structure.

## What makes sense in the current model

- Independent sigmoid gates are appropriate if a payload may contain multiple
  tactics; a softmax would incorrectly force exclusivity.
- A constrained residual `H` is a reasonable place for protocol, syntax, source,
  and other shared information not explained by tactics.
- Golden-label alignment anchors otherwise non-identifiable gate dimensions to named
  tactics.
- Full versus H-only reconstruction and per-condition ablation are useful
  faithfulness checks when combined with semantic held-out metrics.
- Centering/normalizing tactic text embeddings is a reasonable diagnostic geometry
  step, provided raw and transformed similarities are both retained.

## Main modeling problems to correct

1. **Gate values are scores, not calibrated amounts.** A cosine followed by a sharp
   sigmoid and a fixed 0.5 threshold does not yield a probability or physical
   fraction. Report "tactic evidence score" until calibration is measured on held-out
   labels. Multi-label amounts require multi-label targets or another identifiable
   quantitative target.

2. **The real semantic test is empty.** Keep a chronological test for deployment
   realism, but collect/review labels across that time range. As an interim model
   selection check, create a separate stratified golden holdout and never use its
   labels in training; clearly label it as non-temporal validation.

3. **The condition matrix is constant in the encoder.** Concatenating all 13 x 768
   fixed values to every sample adds 9,984 constant features, which act only like
   biases. The residual encoder should consume `x` (and, if needed, sample-specific
   gates), not the same flattened condition table for every row.

4. **The decoder condition input is unnecessarily large.** With fixed conditions,
   the first linear decoder layer applied to flattened `g_i C_i` is algebraically a
   learned transform of the 13 gate scalars. Decode from `[H, gates]`, or first map
   each tactic prototype to a small learned concept vector, to reduce parameters and
   make the bottleneck explicit.

5. **Current auxiliary losses compete before semantic alignment is established.**
   Sparsity pushes gates off, entropy pushes them to extremes, utility asks every
   active gate to meet a large reconstruction margin, and squared-cosine
   decorrelation can penalize valid tactic co-occurrence. Start with reconstruction +
   KL + supervised alignment + a small condition-use constraint. Add one auxiliary
   loss at a time only when an ablation improves held-out semantic and faithfulness
   metrics.

6. **`L_decorrelation` does not disentangle the fixed condition vectors.** Their
   geometry is not learned by this loss; it only penalizes co-activation, and cosine
   squared also penalizes negatively related vectors. This can suppress legitimate
   multi-tactic payloads. Disable it in the feasibility baseline.

7. **Single-label InfoNCE validates only the winning tactic.** It does not supervise
   independent non-target gate calibration or multi-label mixtures. Normal examples
   should be useful all-zero targets, and reviewed multi-tactic rows should use a
   multi-hot BCE-style objective.

8. **Residual leakage needs a stronger test than its training adversary.** The current
   adversary is linear and shares training dynamics with the encoder. Fit a frozen,
   post-hoc nonlinear probe on `H`; tactic predictability should be near a declared
   baseline while gate-space prediction remains useful.

9. **Ablating gates or setting `H=0` is out of distribution.** Retain these metrics,
   but also train with occasional condition/residual dropout so the auxiliary paths
   are observed during training, and compare against shuffled-gate ablations.

## Recommended next milestone

1. Finish/rebuild `x.npy` and verify its row count and manifest atomically.
2. Produce a label coverage table by tactic and time; acquire a real chronological
   test set with enough examples for the tactics in scope.
3. Restrict the first feasibility run to tactics with adequate train and test support
   (or use a clearly defined `other/unknown` policy).
4. Train a minimal baseline: payload projector + supervised tactic alignment, then
   add `H` and reconstruction. Compare against a plain classifier on the same fixed
   payload embeddings.
5. Declare success only if held-out macro F1/AUPRC beats the plain/prevalence
   baselines, gates are calibrated, full reconstruction beats H-only/shuffled-gate
   reconstruction, and a post-hoc probe finds limited tactic leakage in `H`.
6. Only after that result, tune sparsity, adversarial leakage, utility margins, and
   condition geometry.

The default config now implements the first joint feasibility baseline as
`feasibility_baseline_v1`: reconstruction `1.0`, KL `1.0`, behavior InfoNCE `1.0`,
and residual/condition-use constraint `0.1`; all other auxiliary loss weights are
zero. This is the baseline to run once the embedding cache and real labeled
validation split are available.

## Commands

```powershell
uv run python -m experiments.disentangled_cvae_step1.concept_validation `
  --output outputs\disentangled_cvae_step1\concept_validation.json

uv run python -m unittest discover `
  -s experiments\disentangled_cvae_step1\tests -v
```
