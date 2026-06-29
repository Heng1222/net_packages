from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data.adapters import load_raw_data
from .data.condition_loader import ConditionSet, load_condition_set, make_condition_embedder
from .data.dataset import DatasetBundle, SplitIndices
from .data.embedders import load_or_embed_payloads
from .data.preprocessing import fit_transform_splits
from .data.split import make_split
from .evaluation.compatibility_test import run_compatibility_test
from .evaluation.latent import export_latent
from .evaluation.metrics import evaluate_classifier, evaluate_clustering
from .evaluation.visualization import plot_confusion, plot_history, visualize_latent
from .models.ae import AutoEncoder
from .models.cvae import ConditionalVAE
from .training.common import load_best_state
from .training.train_ae import extract_ae_latent, train_ae
from .training.train_cvae import extract_cvae_latent, train_cvae
from .utils.config import save_config
from .utils.io import write_json
from .utils.report import write_report
from .utils.seed import seed_everything


class ExperimentRunner:
    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        device: torch.device,
        reuse_checkpoints: bool = False,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.device = device
        self.reuse_checkpoints = reuse_checkpoints
        self.seed = int(config.get("seed", 42))
        self.logger = logging.getLogger("ae_cvae_tactic")
        self.results: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.condition_sets: dict[str, ConditionSet] = {}
        self.cvae_cache: dict[str, tuple[ConditionalVAE, ConditionSet, dict[str, Any]]] = {}
        self.condition_embedder: Any = None
        self._prepare_data()

    def _prepare_data(self) -> None:
        seed_everything(self.seed)
        data_config = self.config["data"]
        self.logger.info("Loading input data: %s", data_config["input_path"])
        loaded = load_raw_data(data_config)
        embedding_metadata: dict[str, Any]
        if loaded.features is None:
            if loaded.texts is None:
                raise ValueError("Data adapter produced neither features nor payload text.")
            self.logger.info("Embedding %d payload texts on %s", len(loaded.texts), self.device)
            features, embedding_metadata = load_or_embed_payloads(
                loaded.texts, data_config["embedder"], self.device
            )
            duplicate_rows = len(loaded.texts) - len(set(loaded.texts))
            if duplicate_rows:
                self.warnings.append(
                    f"Row-wise splitting found {duplicate_rows} duplicate payload rows beyond the first occurrence; "
                    "identical payloads may cross train/val/test."
                )
        else:
            features = loaded.features
            embedding_metadata = {"source": "precomputed", "output_dim": int(features.shape[1])}
        self.bundle = DatasetBundle(
            features,
            loaded.sample_ids,
            loaded.labels,
            loaded.condition_keys if loaded.condition_keys is not None else loaded.labels,
            loaded.metadata,
        )
        self.split = make_split(self.bundle, data_config["split"], self.seed)
        self.warnings.extend(self.split.warnings)
        self.split.as_assignment(self.bundle.sample_ids, self.bundle.labels).to_csv(
            self.run_dir / "split_assignments.csv", index=False
        )
        self.x, self.transformer = fit_transform_splits(
            self.bundle.x,
            self.split,
            self.config["preprocessing"]["normalization"],
            str(self.run_dir / "scalers" / "scaler.pkl"),
        )
        self._validate_dimensions()
        write_json(embedding_metadata, self.run_dir / "embeddings" / "payload_embedding_metadata.json")
        self.cvae_split = self._make_cvae_split()
        self.data_summary = self._data_summary()
        write_json(self.data_summary, self.run_dir / "metrics" / "data_summary.json")
        save_config(self.config, self.run_dir / "config_resolved.yaml")
        self.logger.info("Data ready: samples=%d, input_dim=%d", len(self.x), self.x.shape[1])

    def _validate_dimensions(self) -> None:
        actual = int(self.x.shape[1])
        for name in ("ae", "cvae"):
            configured = self.config["model"][name].get("input_dim")
            if configured is not None and int(configured) != actual:
                raise ValueError(f"model.{name}.input_dim={configured} does not match inferred input_dim={actual}.")
            self.config["model"][name]["input_dim"] = actual

    def _make_cvae_split(self) -> SplitIndices:
        keys = self.bundle.condition_keys
        mode = self.config["conditions"].get("condition_mode", "full")
        if keys is None and mode != "none":
            raise ValueError(
                "CVAE condition mode requires data.condition_key_col or data.label_col. "
                "With unlabeled/unconditioned data, use conditions.condition_mode=none."
            )
        exclude = set(map(str, self.config["data"].get("exclude_from_cvae", []) or []))
        if keys is None:
            allowed = np.ones(len(self.x), dtype=bool)
        else:
            allowed = ~np.isin(keys.astype(str), list(exclude))
        result = SplitIndices(
            self.split.train[allowed[self.split.train]],
            self.split.val[allowed[self.split.val]],
            self.split.test[allowed[self.split.test]],
            [],
        )
        if len(result.train) == 0 or len(result.test) == 0:
            raise ValueError(
                "CVAE has an empty train or test split after applying data.exclude_from_cvae."
            )
        return result

    def _data_summary(self) -> dict[str, Any]:
        labels = self.bundle.labels
        return {
            "num_samples": len(self.x),
            "input_dim": int(self.x.shape[1]),
            "split_counts": {
                "train": len(self.split.train), "val": len(self.split.val), "test": len(self.split.test)
            },
            "cvae_split_counts": {
                "train": len(self.cvae_split.train), "val": len(self.cvae_split.val), "test": len(self.cvae_split.test)
            },
            "label_counts": (
                dict(pd.Series(labels).value_counts().sort_index().items()) if labels is not None else None
            ),
        }

    def _condition_set(self, mode: str) -> ConditionSet:
        if mode in self.condition_sets:
            return self.condition_sets[mode]
        keys = self.bundle.condition_keys
        if keys is None:
            labels = ["__none__"]
        else:
            selected = np.concatenate((self.cvae_split.train, self.cvae_split.val, self.cvae_split.test))
            labels = sorted(np.unique(keys[selected].astype(str)).tolist())
        if mode not in {"random", "none"} and self.condition_embedder is None:
            self.condition_embedder = make_condition_embedder(self.config["conditions"], self.device)
        condition_set = load_condition_set(
            self.config["conditions"], labels, mode, self.seed, self.condition_embedder
        )
        configured = self.config["model"]["cvae"].get("condition_dim")
        if configured is not None and int(configured) != condition_set.dimension:
            raise ValueError(
                f"model.cvae.condition_dim={configured} does not match inferred condition_dim={condition_set.dimension}."
            )
        self.config["model"]["cvae"]["condition_dim"] = condition_set.dimension
        self.condition_sets[mode] = condition_set
        np.savez_compressed(
            self.run_dir / "embeddings" / f"conditions_{mode}.npz",
            labels=np.asarray(condition_set.labels, dtype=str),
            matrix=condition_set.matrix,
        )
        write_json(
            {"mode": mode, "mapping": condition_set.mapping, "metadata": condition_set.metadata},
            self.run_dir / "embeddings" / f"conditions_{mode}_metadata.json",
        )
        save_config(self.config, self.run_dir / "config_resolved.yaml")
        return condition_set

    def _export_three(
        self,
        prefix: str,
        latent_by_split: dict[str, np.ndarray],
        split: SplitIndices,
    ) -> None:
        fmt = self.config["output"].get("latent_format", "npz")
        for name, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
            labels = self.bundle.labels[indices] if self.bundle.labels is not None else None
            export_latent(
                self.run_dir / "latent" / f"{prefix}_{name}_latent",
                latent_by_split[name],
                self.bundle.sample_ids[indices],
                labels,
                name,
                fmt,
            )

    def _evaluate_representation(
        self,
        prefix: str,
        latent_by_split: dict[str, np.ndarray],
        split: SplitIndices,
        visualization: bool = True,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        labels = self.bundle.labels
        evaluation = self.config["evaluation"]
        if labels is not None and evaluation.get("run_classification", True):
            classification, _ = evaluate_classifier(
                latent_by_split["train"], labels[split.train],
                latent_by_split["test"], labels[split.test],
                self.config["classifier"], self.run_dir / "checkpoints" / f"classifier_{prefix}.pkl"
            )
            output["classification"] = classification
            plot_confusion(
                classification,
                self.run_dir / "plots" / f"confusion_matrix_{prefix}.png",
                f"{prefix} tactic classification",
            )
        if labels is not None and evaluation.get("run_clustering", True):
            clustering, _ = evaluate_clustering(
                latent_by_split["train"], labels[split.train],
                latent_by_split["test"], labels[split.test], self.seed
            )
            output["clustering"] = clustering
        if visualization and evaluation.get("run_visualization", True):
            output["plots"] = visualize_latent(
                latent_by_split["train"], latent_by_split["test"],
                labels[split.test] if labels is not None else None,
                evaluation.get("visualization_methods", ["pca", "tsne"]),
                self.run_dir / "plots" / prefix,
                self.seed,
                int(evaluation.get("visualization_max_samples", 3000)),
            )
        return output

    def run_ae(self) -> dict[str, Any]:
        self.logger.info("Starting AE baseline")
        seed_everything(self.seed)
        model_config = deepcopy(self.config["model"]["ae"])
        model = AutoEncoder.from_config(model_config)
        checkpoint = self.run_dir / "checkpoints" / "ae.pt"
        if self.reuse_checkpoints and checkpoint.is_file():
            load_best_state(model.to(self.device), checkpoint, self.device)
            history: list[dict[str, float]] = []
            training_summary = {"reused_checkpoint": True}
        else:
            trained = train_ae(
                model, self.x[self.split.train], self.x[self.split.val], self.config["training"],
                model_config, self.device, checkpoint, self.seed
            )
            history = trained.history
            training_summary = {"best_epoch": trained.best_epoch, "best_val_loss": trained.best_val_loss}
            pd.DataFrame(history).to_csv(self.run_dir / "logs" / "ae_history.csv", index=False)
            plot_history(history, self.run_dir / "plots" / "ae_training.png", "AE training")
        latent_by_split: dict[str, np.ndarray] = {}
        reconstruction: dict[str, float] = {}
        for name, indices in (("train", self.split.train), ("val", self.split.val), ("test", self.split.test)):
            latent_by_split[name], reconstruction[name] = extract_ae_latent(
                model, self.x[indices], self.device, int(self.config["training"]["batch_size"])
            )
        self._export_three("ae", latent_by_split, self.split)
        metrics = {"training": training_summary, "reconstruction": reconstruction}
        metrics.update(self._evaluate_representation("ae", latent_by_split, self.split))

        labels = self.bundle.labels
        exclude = set(map(str, self.config["data"].get("exclude_from_cvae", []) or []))
        if labels is not None and exclude and self.config["evaluation"].get("run_classification", True):
            train_mask = ~np.isin(labels[self.split.train].astype(str), list(exclude))
            test_mask = ~np.isin(labels[self.split.test].astype(str), list(exclude))
            fair, _ = evaluate_classifier(
                latent_by_split["train"][train_mask], labels[self.split.train][train_mask],
                latent_by_split["test"][test_mask], labels[self.split.test][test_mask],
                self.config["classifier"], self.run_dir / "checkpoints" / "classifier_ae_mitre_only.pkl"
            )
            metrics["mitre_only_classification"] = fair
            plot_confusion(fair, self.run_dir / "plots" / "confusion_matrix_ae_mitre_only.png", "AE MITRE-only")
            fair_result: dict[str, Any] = {"classification": fair}
            if self.config["evaluation"].get("run_clustering", True):
                fair_clustering, _ = evaluate_clustering(
                    latent_by_split["train"][train_mask], labels[self.split.train][train_mask],
                    latent_by_split["test"][test_mask], labels[self.split.test][test_mask], self.seed
                )
                metrics["mitre_only_clustering"] = fair_clustering
                fair_result["clustering"] = fair_clustering
            self.results["ae_mitre_only"] = fair_result
        write_json(metrics, self.run_dir / "metrics" / "ae_metrics.json")
        self.results["ae"] = metrics
        return metrics

    def _train_cvae_mode(self, mode: str, full_evaluation: bool = True) -> tuple[ConditionalVAE, ConditionSet, dict[str, Any]]:
        if mode in self.cvae_cache:
            return self.cvae_cache[mode]
        self.logger.info("Starting CVAE condition mode: %s", mode)
        seed_everything(self.seed)
        condition_set = self._condition_set(mode)
        keys = self.bundle.condition_keys
        if keys is None:
            keys = np.full(len(self.x), "__none__", dtype=str)
        condition_all = condition_set.for_keys(keys[self._cvae_all_indices()])
        lookup = {index: row for row, index in enumerate(self._cvae_all_indices())}

        def c_for(indices: np.ndarray) -> np.ndarray:
            return condition_all[[lookup[int(index)] for index in indices]]

        model_config = deepcopy(self.config["model"]["cvae"])
        model_config["condition_dim"] = condition_set.dimension
        model = ConditionalVAE.from_config(model_config)
        checkpoint_name = "cvae.pt" if mode == self.config["conditions"].get("condition_mode", "full") else f"cvae_{mode}.pt"
        checkpoint = self.run_dir / "checkpoints" / checkpoint_name
        if self.reuse_checkpoints and checkpoint.is_file():
            load_best_state(model.to(self.device), checkpoint, self.device)
            history: list[dict[str, float]] = []
            training_summary = {"reused_checkpoint": True}
        else:
            trained = train_cvae(
                model,
                self.x[self.cvae_split.train], c_for(self.cvae_split.train),
                self.x[self.cvae_split.val], c_for(self.cvae_split.val),
                self.config["training"], model_config, self.device, checkpoint, self.seed,
            )
            history = trained.history
            training_summary = {"best_epoch": trained.best_epoch, "best_val_loss": trained.best_val_loss}
            pd.DataFrame(history).to_csv(self.run_dir / "logs" / f"cvae_{mode}_history.csv", index=False)
            plot_history(history, self.run_dir / "plots" / f"cvae_{mode}_training.png", f"CVAE {mode} training")
        latent_by_split: dict[str, np.ndarray] = {}
        losses_by_split: dict[str, dict[str, float]] = {}
        for name, indices in (
            ("train", self.cvae_split.train), ("val", self.cvae_split.val), ("test", self.cvae_split.test)
        ):
            seed_everything(self.seed + {"train": 0, "val": 1, "test": 2}[name])
            latent_by_split[name], losses_by_split[name] = extract_cvae_latent(
                model, self.x[indices], c_for(indices), self.device, int(self.config["training"]["batch_size"])
            )
        prefix = "cvae" if mode == self.config["conditions"].get("condition_mode", "full") else f"cvae_{mode}"
        self._export_three(prefix, latent_by_split, self.cvae_split)
        metrics: dict[str, Any] = {"mode": mode, "training": training_summary, "losses": losses_by_split}
        metrics.update(self._evaluate_representation(prefix, latent_by_split, self.cvae_split, visualization=full_evaluation))
        write_json(metrics, self.run_dir / "metrics" / f"{prefix}_metrics.json")
        self.cvae_cache[mode] = (model, condition_set, metrics)
        if mode == self.config["conditions"].get("condition_mode", "full"):
            self.results["cvae"] = metrics
        return model, condition_set, metrics

    def _cvae_all_indices(self) -> np.ndarray:
        return np.concatenate((self.cvae_split.train, self.cvae_split.val, self.cvae_split.test))

    def run_cvae(self) -> dict[str, Any]:
        mode = self.config["conditions"].get("condition_mode", "full")
        return self._train_cvae_mode(mode)[2]

    def run_compatibility(self, mode: str | None = None, prefix: str = "compatibility") -> dict[str, Any]:
        selected_mode = mode or self.config["conditions"].get("condition_mode", "full")
        model, condition_set, _ = self._train_cvae_mode(selected_mode)
        labels = self.bundle.labels[self.cvae_split.test] if self.bundle.labels is not None else None
        result = run_compatibility_test(
            model,
            self.x[self.cvae_split.test],
            self.bundle.sample_ids[self.cvae_split.test],
            labels,
            condition_set,
            self.device,
            int(self.config["training"]["batch_size"]),
            self.run_dir / "metrics",
            prefix,
        )
        if result.get("classification"):
            plot_confusion(
                result["classification"],
                self.run_dir / "plots" / f"confusion_matrix_{prefix}.png",
                f"Compatibility ({selected_mode})",
            )
        if mode is None:
            self.results["compatibility"] = result
        return result

    def run_ablation(self) -> Path:
        rows: list[dict[str, Any]] = []
        for mode in self.config.get("ablation", {}).get(
            "modes", ["full", "short", "keywords", "random", "wrong", "none"]
        ):
            _, _, metrics = self._train_cvae_mode(mode, full_evaluation=False)
            compatibility = self.run_compatibility(mode, f"compatibility_{mode}")
            test_losses = metrics["losses"]["test"]
            classification = metrics.get("classification", {})
            clustering = metrics.get("clustering", {})
            comp_classification = compatibility.get("classification", {})
            rows.append(
                {
                    "condition_mode": mode,
                    "recon_mse": test_losses.get("recon_mse"),
                    "recon_nll": test_losses.get("recon_nll"),
                    "kl_loss": test_losses.get("kl_loss"),
                    "elbo": test_losses.get("elbo"),
                    "negative_elbo": test_losses.get("negative_elbo"),
                    "acc": classification.get("accuracy"),
                    "macro_f1": classification.get("macro_f1"),
                    "compatibility_acc": comp_classification.get("accuracy"),
                    "compatibility_macro_f1": comp_classification.get("macro_f1"),
                    "silhouette": clustering.get("silhouette"),
                    "nmi": clustering.get("nmi"),
                    "ari": clustering.get("ari"),
                }
            )
        path = self.run_dir / "metrics" / "condition_ablation_summary.csv"
        pd.DataFrame(rows).fillna("NA").to_csv(path, index=False)
        self.results["ablation_summary"] = str(path)
        return path

    def finalize_report(self) -> None:
        write_report(
            self.run_dir / "reports" / "report.md",
            self.config,
            self.data_summary,
            self.results,
            self.warnings,
        )
        write_json(self.results, self.run_dir / "metrics" / "run_summary.json")
