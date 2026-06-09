"""BERT-based task difficulty predictor for adaptive task sampling.

Predicts per-task success rate p_hat(x) from task instruction + evaluation
rubrics. Used to filter training to the productive zone [ε, 1-ε] and
prioritize tasks closest to p=0.5 (maximum GRPO signal).
"""

import io
import random
import threading
from typing import Any

import torch
import torch.nn as nn

from areal.utils.logging import getLogger

logger = getLogger("DifficultyPredictor")


class DifficultyPredictor:
    """Small BERT model that predicts task success rate.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path (e.g. "distilbert-base-uncased").
    max_length : int
        Max token length for BERT input.
    device : str
        Device for training and inference.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        device: str = "cpu",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._encoder = AutoModel.from_pretrained(model_name).to(device)
        hidden_size = self._encoder.config.hidden_size
        self._head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        ).to(device)
        self._lock = threading.Lock()
        self.last_selection_stats: dict[str, float] = {}

    def train_on_data(
        self,
        texts: list[str],
        targets: list[float],
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 32,
        max_samples: int = 5000,
    ) -> dict[str, float]:
        """Fine-tune on collected (text, pass_rate) pairs.

        Returns dict with training stats (mse, n_samples).
        Acquires self._lock so concurrent predict() calls (from rollout
        controller's task_input_generator) wait until training finishes.
        """
        if len(texts) > max_samples:
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            targets = [targets[i] for i in indices]

        with self._lock:
            self._encoder.train()
            self._head.train()
            optimizer = torch.optim.AdamW(
                list(self._encoder.parameters()) + list(self._head.parameters()),
                lr=lr,
            )
            loss_fn = nn.MSELoss()
            target_tensor = torch.tensor(
                targets, dtype=torch.float32, device=self.device
            )

            total_loss = 0.0
            n_batches = 0
            for _epoch in range(epochs):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_targets = target_tensor[i : i + batch_size]

                    enc = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    hidden = self._encoder(**enc).last_hidden_state[:, 0]  # [CLS]
                    pred = self._head(hidden).squeeze(-1)
                    loss = loss_fn(pred, batch_targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

            self._encoder.eval()
            self._head.eval()

        avg_mse = total_loss / max(n_batches, 1)
        logger.info(
            "[train] n_samples=%d epochs=%d avg_mse=%.4f",
            len(texts), epochs, avg_mse,
        )
        return {"mse": avg_mse, "n_samples": len(texts)}

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 64) -> list[float]:
        """Batch inference, returns p_hat for each text.

        Acquires self._lock so a concurrent train_on_data() (run in the
        trainer's background thread) does not race on model parameters.
        """
        with self._lock:
            self._encoder.eval()
            self._head.eval()
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self._encoder(**enc).last_hidden_state[:, 0]
                pred = self._head(hidden).squeeze(-1)
                results.extend(pred.cpu().tolist())
        return results

    def select_tasks(
        self,
        all_tasks: list[dict[str, Any]],
        sample_size: int = 1024,
        select_top_k: int | None = None,
        epsilon: float = 0.1,
        target_p: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Sample tasks and select those closest to p=target_p.

        Parameters
        ----------
        all_tasks : list of task dicts (must have "task_name", "evaluator_reference")
        sample_size : how many to randomly sample for scoring
        select_top_k : how many to keep (default: sample_size // 4)
        epsilon : filter threshold ([ε, 1-ε])
        target_p : predicted pass-rate the selector is pulling toward
        """
        if select_top_k is None:
            select_top_k = max(1, sample_size // 4)

        # Uniform sample across difficulty levels. The raw dataset is
        # dominated by diff 1-3 (~79%), so proportional sampling starves
        # the productive zone at higher difficulties as the policy improves.
        from collections import defaultdict as _dd
        by_diff: dict[int, list] = _dd(list)
        for t in all_tasks:
            try:
                d = int(t.get("difficulty", 5))
            except (TypeError, ValueError):
                d = 5
            by_diff[d].append(t)

        if len(all_tasks) <= sample_size:
            sampled = list(all_tasks)
        else:
            n_levels = len(by_diff)
            per_level = max(1, sample_size // n_levels)
            sampled = []
            leftover_pool = []
            for _d, tasks_d in by_diff.items():
                if len(tasks_d) <= per_level:
                    sampled.extend(tasks_d)
                else:
                    picked = random.sample(tasks_d, per_level)
                    sampled.extend(picked)
                    picked_ids = {id(t) for t in picked}
                    leftover_pool.extend(
                        t for t in tasks_d if id(t) not in picked_ids
                    )
            # Top up to sample_size from remaining pool (uniform over pool)
            if len(sampled) < sample_size and leftover_pool:
                remaining = sample_size - len(sampled)
                extra = random.sample(
                    leftover_pool, min(remaining, len(leftover_pool))
                )
                sampled.extend(extra)

        # BERT inference
        texts = [self.format_input(t) for t in sampled]
        p_hats = self.predict(texts)

        # Filter to [ε, 1-ε] and sort by |p_hat - target_p|
        scored = [
            (task, p)
            for task, p in zip(sampled, p_hats)
            if epsilon <= p <= 1 - epsilon
        ]
        scored.sort(key=lambda x: abs(x[1] - target_p))

        selected = [task for task, _ in scored[:select_top_k]]

        # Collect difficulty stats for selected tasks
        difficulties = [
            int(t.get("difficulty", 5)) for t in selected
        ]
        if difficulties:
            self.last_selection_stats = {
                "difficulty_avg": sum(difficulties) / len(difficulties),
                "difficulty_min": min(difficulties),
                "difficulty_max": max(difficulties),
                "p_hat_avg": sum(p for _, p in scored[:select_top_k]) / max(len(scored[:select_top_k]), 1),
                "in_zone": len(scored),
                "n_selected": len(selected),
            }
        logger.info(
            "[select_tasks] sampled=%d in_zone=%d selected=%d "
            "diff=%.1f/[%d,%d]",
            len(sampled), len(scored), len(selected),
            self.last_selection_stats.get("difficulty_avg", 0),
            self.last_selection_stats.get("difficulty_min", 0),
            self.last_selection_stats.get("difficulty_max", 0),
        )
        return selected

    def state_dict_bytes(self) -> bytes:
        """Serialize model to bytes for distribution."""
        buf = io.BytesIO()
        state = {
            "encoder": self._encoder.state_dict(),
            "head": self._head.state_dict(),
        }
        torch.save(state, buf)
        return buf.getvalue()

    def load_state_dict_bytes(self, data: bytes):
        """Load weights from serialized bytes."""
        buf = io.BytesIO(data)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._encoder.load_state_dict(state["encoder"])
        self._head.load_state_dict(state["head"])
        self._encoder.eval()
        self._head.eval()

    @staticmethod
    def format_input(task: dict[str, Any]) -> str:
        """Convert task fields to BERT input string.

        Includes the dataset-provided ``difficulty`` integer (1-15) as a
        prefix token so BERT has a direct signal for success-rate rather
        than having to infer it from free-form task text alone. Without
        this the model collapses to predicting the population mean (~0.15
        reward) across all difficulties — training MSE ≈ Var(y), p_hat
        barely discriminates, and the adaptive selector that sorts by
        |p_hat - target_p| ends up near-random. Prefixing the integer
        gives BERT a trivial correlation channel; the rest of the text
        is still free to refine within-difficulty distinctions.
        """
        import json as _json

        name = task.get("task_name", "")
        ref = task.get("evaluator_reference", "")
        diff_raw = task.get("difficulty", "")
        try:
            diff_int = int(diff_raw)
        except (TypeError, ValueError):
            diff_int = None
        # evaluator_reference may be a JSON string, list of dicts, or plain string
        if isinstance(ref, str):
            try:
                ref = _json.loads(ref)
            except (ValueError, TypeError):
                pass  # keep as plain string
        if isinstance(ref, list):
            parts = []
            for item in ref:
                if isinstance(item, dict):
                    desc = item.get("description", "")
                    facts = item.get("facts", [])
                    if desc:
                        parts.append(desc)
                    if facts:
                        parts.extend(str(f) for f in facts)
                else:
                    parts.append(str(item))
            ref_text = " ".join(parts)
        else:
            ref_text = str(ref)
        diff_prefix = f"[DIFF={diff_int}] " if diff_int is not None else ""
        return f"{diff_prefix}{name} [SEP] {ref_text}"
