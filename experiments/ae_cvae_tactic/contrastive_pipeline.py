from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .evaluation.compatibility_test import run_compatibility_test
from .evaluation.metrics import classification_metrics
from .evaluation.visualization import plot_confusion, plot_history
from .models.contrastive_cvae import ContrastiveConditionalVAE
from .pipeline import ExperimentRunner
from .training.common import load_best_state
from .training.train_contrastive_cvae import (
    extract_contrastive_cvae,
    train_contrastive_cvae,
)
from .utils.config import save_config
from .utils.io import write_json
from .utils.seed import seed_everything


class ContrastiveExperimentRunner(ExperimentRunner):
    """Opt-in experiment runner; the original ExperimentRunner is unchanged."""

    def _contrastive_config(self, condition_dim: int) -> dict[str, Any]:
        model_section = self.config.get("model", {}).get("contrastive_cvae")
        if not isinstance(model_section, dict):
            raise ValueError("model.contrastive_cvae is required for the contrastive experiment.")
        model_config = deepcopy(model_section)
        model_config["input_dim"] = int(self.x.shape[1])
        model_config["condition_dim"] = condition_dim
        if model_config.get("projection_dim") is None:
            model_config["projection_dim"] = condition_dim
        if model_config.get("condition_projection", "identity") != "identity":
            raise ValueError("condition_projection must be 'identity' for this experiment.")
        if int(model_config["projection_dim"]) != condition_dim:
            raise ValueError(
                "The frozen identity condition branch requires projection_dim == condition_dim."
            )
        self.config["model"]["contrastive_cvae"] = model_config
        save_config(self.config, self.run_dir / "config_resolved.yaml")
        return model_config

    @staticmethod
    def _target_indices(keys: np.ndarray, labels: list[str]) -> np.ndarray:
        lookup = {label: index for index, label in enumerate(labels)}
        missing = sorted({str(key) for key in keys if str(key) not in lookup})
        if missing:
            raise KeyError(f"Contrastive targets are missing condition labels: {missing}")
        return np.asarray([lookup[str(key)] for key in keys], dtype=np.int64)

    def run_contrastive(self) -> dict[str, Any]:
        self.logger.info("Starting opt-in contrastive CVAE experiment")
        seed_everything(self.seed)
        mode = self.config["conditions"].get("condition_mode", "full")
        if mode == "none":
            raise ValueError("Contrastive training requires semantic or ablation condition candidates, not 'none'.")
        condition_set = self._condition_set(mode)
        if len(condition_set.labels) < 2:
            raise ValueError("Contrastive training requires at least two candidate conditions.")
        keys = self.bundle.condition_keys
        if keys is None:
            raise ValueError("Contrastive training requires data.condition_key_col or data.label_col.")

        model_config = self._contrastive_config(condition_set.dimension)
        model = ContrastiveConditionalVAE.from_config(model_config)
        candidates = condition_set.matrix

        def arrays(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            selected_keys = keys[indices].astype(str)
            return (
                condition_set.for_keys(selected_keys),
                self._target_indices(selected_keys, condition_set.labels),
            )

        c_train, target_train = arrays(self.cvae_split.train)
        c_val, target_val = arrays(self.cvae_split.val)
        checkpoint = self.run_dir / "checkpoints" / "contrastive_cvae.pt"
        if self.reuse_checkpoints and checkpoint.is_file():
            load_best_state(model.to(self.device), checkpoint, self.device)
            history: list[dict[str, float]] = []
            training_summary: dict[str, Any] = {"reused_checkpoint": True}
        else:
            trained = train_contrastive_cvae(
                model,
                self.x[self.cvae_split.train],
                c_train,
                target_train,
                self.x[self.cvae_split.val],
                c_val,
                target_val,
                candidates,
                self.config["training"],
                model_config,
                self.device,
                checkpoint,
                self.seed,
            )
            history = trained.history
            training_summary = {
                "best_epoch": trained.best_epoch,
                "best_val_loss": trained.best_val_loss,
            }
            pd.DataFrame(history).to_csv(
                self.run_dir / "logs" / "contrastive_cvae_history.csv", index=False
            )
            plot_history(
                history,
                self.run_dir / "plots" / "contrastive_cvae_training.png",
                "Contrastive CVAE training",
            )

        latent_by_split: dict[str, np.ndarray] = {}
        projection_by_split: dict[str, np.ndarray] = {}
        logits_by_split: dict[str, np.ndarray] = {}
        losses_by_split: dict[str, dict[str, float]] = {}
        targets_by_split: dict[str, np.ndarray] = {}
        split_pairs = (
            ("train", self.cvae_split.train),
            ("val", self.cvae_split.val),
            ("test", self.cvae_split.test),
        )
        for offset, (name, indices) in enumerate(split_pairs):
            seed_everything(self.seed + offset)
            conditions, targets = arrays(indices)
            latent, projection, logits, losses = extract_contrastive_cvae(
                model,
                self.x[indices],
                conditions,
                targets,
                candidates,
                self.device,
                int(self.config["training"]["batch_size"]),
            )
            latent_by_split[name] = latent
            projection_by_split[name] = projection
            logits_by_split[name] = logits
            losses_by_split[name] = losses
            targets_by_split[name] = targets

        self._export_three("contrastive_cvae", latent_by_split, self.cvae_split)
        self._export_three("contrastive_payload", projection_by_split, self.cvae_split)

        test_indices = self.cvae_split.test
        true_labels = keys[test_indices].astype(str)
        predicted_indices = logits_by_split["test"].argmax(axis=1)
        predicted_labels = np.asarray(
            [condition_set.labels[index] for index in predicted_indices], dtype=str
        )
        retrieval = classification_metrics(
            true_labels, predicted_labels, condition_set.labels
        )
        plot_confusion(
            retrieval,
            self.run_dir / "plots" / "confusion_matrix_contrastive_retrieval.png",
            "Payload-to-description contrastive retrieval",
        )
        np.savez_compressed(
            self.run_dir / "metrics" / "contrastive_score_matrix.npz",
            sample_id=self.bundle.sample_ids[test_indices].astype(str),
            true_label=true_labels,
            candidate_labels=np.asarray(condition_set.labels, dtype=str),
            logits=logits_by_split["test"],
            cosine_similarity=logits_by_split["test"] * model.temperature,
        )
        result_frame = pd.DataFrame(
            {
                "sample_id": self.bundle.sample_ids[test_indices].astype(str),
                "true_label": true_labels,
                "predicted_label": predicted_labels,
                "correct": predicted_labels == true_labels,
            }
        )
        for index, label in enumerate(condition_set.labels):
            result_frame[f"score::{label}"] = logits_by_split["test"][:, index]
        result_frame.to_csv(
            self.run_dir / "metrics" / "contrastive_retrieval_results.csv", index=False
        )

        payload_evaluation = self._evaluate_representation(
            "contrastive_payload", projection_by_split, self.cvae_split
        )
        oracle_latent_evaluation = self._evaluate_representation(
            "contrastive_cvae", latent_by_split, self.cvae_split
        )
        compatibility = run_compatibility_test(
            model,
            self.x[test_indices],
            self.bundle.sample_ids[test_indices],
            self.bundle.labels[test_indices] if self.bundle.labels is not None else None,
            condition_set,
            self.device,
            int(self.config["training"]["batch_size"]),
            self.run_dir / "metrics",
            "contrastive_reconstruction_compatibility",
        )
        if compatibility.get("classification"):
            plot_confusion(
                compatibility["classification"],
                self.run_dir
                / "plots"
                / "confusion_matrix_contrastive_reconstruction_compatibility.png",
                "Contrastive CVAE reconstruction compatibility",
            )

        metrics = {
            "mode": mode,
            "semantic_constraint": {
                "payload_branch_inputs": ["payload_embedding"],
                "condition_branch": "frozen_identity",
                "candidate_count": len(condition_set.labels),
                "projection_dim": model.projection_dim,
                "temperature": model.temperature,
                "contrastive_weight": model.contrastive_weight,
            },
            "training": training_summary,
            "losses": losses_by_split,
            "retrieval": retrieval,
            "payload_projection": payload_evaluation,
            "oracle_cvae_latent": oracle_latent_evaluation,
            "reconstruction_compatibility": compatibility,
        }
        write_json(metrics, self.run_dir / "metrics" / "contrastive_cvae_metrics.json")
        self.results["contrastive_cvae"] = metrics
        return metrics

    def finalize_contrastive_report(self) -> None:
        metrics = self.results.get("contrastive_cvae", {})
        retrieval = metrics.get("retrieval", {})
        test_losses = metrics.get("losses", {}).get("test", {})
        config = self.config.get("model", {}).get("contrastive_cvae", {})
        lines = [
            "# Contrastive CVAE Experiment Report",
            "",
            "This is an opt-in experiment. The original AE/CVAE pipeline and checkpoints are not modified.",
            "",
            "## Configuration",
            "",
            f"- Condition mode: {self.config.get('conditions', {}).get('condition_mode')}",
            f"- Temperature: {config.get('temperature')}",
            f"- Contrastive weight: {config.get('contrastive_weight')}",
            f"- Projection dimension: {config.get('projection_dim')}",
            "- Condition projection: frozen identity",
            "- Contrastive payload input: payload only (oracle condition is excluded)",
            "",
            "## Test results",
            "",
            f"- Retrieval accuracy: {retrieval.get('accuracy')}",
            f"- Retrieval macro F1: {retrieval.get('macro_f1')}",
            f"- Retrieval weighted F1: {retrieval.get('weighted_f1')}",
            f"- Contrastive loss: {test_losses.get('contrastive_loss')}",
            f"- Negative ELBO: {test_losses.get('negative_elbo')}",
            f"- Total loss: {test_losses.get('total_loss')}",
            "",
            "## Interpretation boundary",
            "",
            "The fixed nine-description experiment tests alignment to semantic prototypes. It can still learn a closed-set nine-class mapping. Strong semantic evidence additionally requires held-out paraphrases or unseen descriptions.",
            "",
        ]
        (self.run_dir / "reports" / "contrastive_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        write_json(self.results, self.run_dir / "metrics" / "run_summary.json")
