"""Track per-task empirical success rates for adaptive task sampling.

Used to filter out tasks where p ≈ 0 (all-fail) or p ≈ 1 (all-success),
keeping training focused on the productive zone [ε, 1-ε].
"""

import threading

from areal.utils.logging import getLogger

logger = getLogger("TaskSuccessTracker")


class TaskSuccessTracker:
    """Exponential moving average tracker for per-task success rates.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor. Higher = more weight on recent observations.
    min_samples : int
        Minimum observations before a task's rate is considered reliable.
        Tasks with fewer samples are always included (exploration).
    epsilon : float
        Tasks with p_hat < epsilon or p_hat > 1-epsilon are filtered out.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        min_samples: int = 2,
        epsilon: float = 0.1,
    ):
        self._rates: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._texts: dict[str, str] = {}  # task_id -> formatted text for BERT
        self._all_observations: list[tuple[str, float]] = []  # (text, pass_rate)
        self._alpha = alpha
        self._min_samples = min_samples
        self.epsilon = epsilon
        self._lock = threading.Lock()

    def update(
        self,
        task_id: str,
        n_success: int,
        n_total: int,
        task_text: str = "",
    ):
        """Update success rate for a task from a GRPO group outcome."""
        if not task_id or n_total <= 0:
            return
        success_rate = n_success / n_total
        with self._lock:
            count = self._counts.get(task_id, 0) + 1
            self._counts[task_id] = count
            old = self._rates.get(task_id, 0.5)
            self._rates[task_id] = old * (1 - self._alpha) + success_rate * self._alpha
            if task_text:
                self._texts[task_id] = task_text
                self._all_observations.append((task_text, success_rate))

    def get_rate(self, task_id: str) -> float | None:
        """Get estimated success rate, or None if insufficient data."""
        with self._lock:
            if self._counts.get(task_id, 0) < self._min_samples:
                return None
            return self._rates.get(task_id)

    def should_sample(self, task_id: str) -> bool:
        """Whether this task is in the productive zone [ε, 1-ε]."""
        p = self.get_rate(task_id)
        if p is None:
            return True  # Unknown tasks: always sample (explore)
        return self.epsilon <= p <= 1 - self.epsilon

    def sampling_priority(self, task_id: str) -> float:
        """Priority weight: higher for tasks closer to p=0.5."""
        p = self.get_rate(task_id)
        if p is None:
            return 1.0  # Unknown tasks get full priority
        if p < self.epsilon or p > 1 - self.epsilon:
            return 0.0  # Outside productive zone
        # Proportional to p(1-p), peaked at p=0.5
        return 4.0 * p * (1 - p)

    def get_training_data(self) -> tuple[list[str], list[float]]:
        """Export all (text, pass_rate) observations for BERT training."""
        with self._lock:
            if not self._all_observations:
                return [], []
            texts, targets = zip(*self._all_observations)
            return list(texts), list(targets)

    def stats(self) -> dict:
        """Summary statistics for logging."""
        with self._lock:
            n_tracked = len(self._rates)
            n_reliable = sum(
                1 for tid in self._rates if self._counts.get(tid, 0) >= self._min_samples
            )
            if not self._rates:
                return {"n_tracked": 0, "n_reliable": 0}
            rates = [
                self._rates[tid]
                for tid in self._rates
                if self._counts.get(tid, 0) >= self._min_samples
            ]
            n_filtered = sum(
                1 for r in rates if r < self.epsilon or r > 1 - self.epsilon
            )
            return {
                "n_tracked": n_tracked,
                "n_reliable": n_reliable,
                "n_filtered": n_filtered,
                "mean_rate": sum(rates) / len(rates) if rates else 0,
            }
