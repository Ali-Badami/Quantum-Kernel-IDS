"""
D-optimal sensor selection using greedy submodular maximization.

Selects sensors that maximize the log-determinant of the kernel submatrix,
which corresponds to D-optimal experimental design.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import heapq
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SensorSelector:
    """
    Greedy sensor selection for D-optimal design.

    Selects a subset of sensors that maximizes the log-determinant of the
    kernel (correlation) matrix restricted to those sensors. Uses lazy
    greedy evaluation for efficiency.

    Attributes:
        regularization: Regularization parameter for matrix inversion
    """

    def __init__(self, regularization: float = 1e-6):
        """
        Initialize the sensor selector.

        Args:
            regularization: Small value added to diagonal for numerical stability
        """
        self.regularization = regularization

    def _log_det(self, K: np.ndarray, indices: List[int]) -> float:
        """
        Compute log-determinant of kernel submatrix.

        Args:
            K: Full kernel/correlation matrix
            indices: Indices of selected sensors

        Returns:
            Log-determinant value
        """
        if len(indices) == 0:
            return float("-inf")

        K_sub = K[np.ix_(indices, indices)]
        K_reg = K_sub + self.regularization * np.eye(len(indices))

        try:
            sign, logdet = np.linalg.slogdet(K_reg)
            if sign <= 0:
                return float("-inf")
            return logdet
        except np.linalg.LinAlgError:
            return float("-inf")

    def _marginal_gain(
        self, K: np.ndarray, current: List[int], candidate: int
    ) -> float:
        """
        Compute marginal gain from adding a sensor.

        Args:
            K: Full kernel/correlation matrix
            current: Currently selected sensor indices
            candidate: Index of candidate sensor to add

        Returns:
            Marginal increase in log-determinant
        """
        current_val = self._log_det(K, current)
        new_val = self._log_det(K, current + [candidate])
        return new_val - current_val

    def select_greedy(
        self, K: np.ndarray, k: int, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select k sensors using standard greedy algorithm.

        At each iteration, selects the sensor with maximum marginal gain.
        Time complexity: O(k * n * n^3) where n is the number of sensors.

        Args:
            K: Kernel/correlation matrix of shape (n_sensors, n_sensors)
            k: Number of sensors to select
            feature_names: Optional list of sensor names

        Returns:
            Dictionary with selection results
        """
        n_sensors = K.shape[0]
        if k > n_sensors:
            raise ValueError(f"k={k} exceeds number of sensors ({n_sensors})")

        selected = []
        remaining = list(range(n_sensors))
        objective_values = []

        logger.info(f"Selecting {k} sensors from {n_sensors} using greedy algorithm")

        for iteration in range(k):
            best_gain = float("-inf")
            best_idx = None

            for idx in remaining:
                gain = self._marginal_gain(K, selected, idx)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx is None:
                logger.warning(f"Could not find valid sensor at iteration {iteration}")
                break

            selected.append(best_idx)
            remaining.remove(best_idx)
            objective_values.append(self._log_det(K, selected))

            if feature_names:
                logger.debug(f"Selected sensor {iteration+1}: {feature_names[best_idx]} "
                           f"(gain={best_gain:.4f})")

        final_objective = self._log_det(K, selected)

        result = {
            "selected_indices": selected,
            "objective_value": float(final_objective),
            "objective_history": [float(v) for v in objective_values],
            "k": k,
            "n_sensors": n_sensors,
        }

        if feature_names:
            result["selected_names"] = [feature_names[i] for i in selected]

        return result

    def select_lazy_greedy(
        self, K: np.ndarray, k: int, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select k sensors using lazy greedy algorithm.

        Uses a priority queue to avoid recomputing all marginal gains at
        each iteration. Much faster than standard greedy for large n.
        Time complexity: O(k * n * n^3) worst case, but typically O(k * n^3).

        Args:
            K: Kernel/correlation matrix of shape (n_sensors, n_sensors)
            k: Number of sensors to select
            feature_names: Optional list of sensor names

        Returns:
            Dictionary with selection results
        """
        n_sensors = K.shape[0]
        if k > n_sensors:
            raise ValueError(f"k={k} exceeds number of sensors ({n_sensors})")

        selected = []
        objective_values = []

        # Initialize priority queue with upper bounds (negative for max-heap behavior)
        # Format: (-upper_bound, index, is_current)
        heap = [(-float("inf"), i, False) for i in range(n_sensors)]
        heapq.heapify(heap)

        logger.info(f"Selecting {k} sensors from {n_sensors} using lazy greedy")

        for iteration in range(k):
            while True:
                neg_bound, idx, is_current = heapq.heappop(heap)

                if idx in selected:
                    continue

                if is_current:
                    # This gain is up-to-date, select this sensor
                    selected.append(idx)
                    objective_values.append(self._log_det(K, selected))
                    break
                else:
                    # Recompute gain and push back
                    new_gain = self._marginal_gain(K, selected, idx)
                    heapq.heappush(heap, (-new_gain, idx, True))

        final_objective = self._log_det(K, selected)

        result = {
            "selected_indices": selected,
            "objective_value": float(final_objective),
            "objective_history": [float(v) for v in objective_values],
            "k": k,
            "n_sensors": n_sensors,
            "algorithm": "lazy_greedy",
        }

        if feature_names:
            result["selected_names"] = [feature_names[i] for i in selected]

        return result


def variance_baseline(X: np.ndarray, k: int) -> List[int]:
    """
    Select top-k sensors by variance.

    Simple baseline that selects sensors with highest variance,
    assuming high-variance sensors are more informative.

    Args:
        X: Data array of shape (n_samples, n_sensors)
        k: Number of sensors to select

    Returns:
        List of selected sensor indices
    """
    variances = np.var(X, axis=0)
    return list(np.argsort(variances)[-k:][::-1])


def correlation_baseline(
    X: np.ndarray, k: int, regularization: float = 1e-6
) -> List[int]:
    """
    Select sensors using classical correlation-based greedy.

    Uses Pearson correlation matrix instead of quantum kernel.

    Args:
        X: Data array of shape (n_samples, n_sensors)
        k: Number of sensors to select
        regularization: Regularization for log-det computation

    Returns:
        List of selected sensor indices
    """
    corr = np.corrcoef(X.T)
    selector = SensorSelector(regularization=regularization)
    result = selector.select_lazy_greedy(corr, k)
    return result["selected_indices"]


def random_baseline(n_sensors: int, k: int, seed: int = 42) -> List[int]:
    """
    Randomly select k sensors.

    Baseline for comparison - any method should outperform random selection.

    Args:
        n_sensors: Total number of sensors
        k: Number of sensors to select
        seed: Random seed for reproducibility

    Returns:
        List of randomly selected sensor indices
    """
    rng = np.random.RandomState(seed)
    return list(rng.choice(n_sensors, size=k, replace=False))


def compare_methods(
    K_quantum: np.ndarray,
    X: np.ndarray,
    k_values: List[int],
    feature_names: Optional[List[str]] = None,
    regularization: float = 1e-6,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Compare quantum and baseline selection methods.

    Args:
        K_quantum: Quantum kernel/correlation matrix
        X: Original data for baseline methods
        k_values: List of k values to evaluate
        feature_names: Optional sensor names
        regularization: Regularization parameter

    Returns:
        Nested dict: method -> k -> results
    """
    selector = SensorSelector(regularization=regularization)
    results = {"quantum": {}, "variance": {}, "correlation": {}, "random": {}}

    for k in k_values:
        logger.info(f"Comparing methods for k={k}")

        # Quantum kernel selection
        results["quantum"][k] = selector.select_lazy_greedy(K_quantum, k, feature_names)

        # Variance baseline
        var_indices = variance_baseline(X, k)
        results["variance"][k] = {
            "selected_indices": var_indices,
            "objective_value": float(selector._log_det(K_quantum, var_indices)),
        }

        # Correlation baseline
        corr_indices = correlation_baseline(X, k, regularization)
        results["correlation"][k] = {
            "selected_indices": corr_indices,
            "objective_value": float(selector._log_det(K_quantum, corr_indices)),
        }

        # Random baseline
        rand_indices = random_baseline(X.shape[1], k)
        results["random"][k] = {
            "selected_indices": rand_indices,
            "objective_value": float(selector._log_det(K_quantum, rand_indices)),
        }

        if feature_names:
            for method in results:
                results[method][k]["selected_names"] = [
                    feature_names[i] for i in results[method][k]["selected_indices"]
                ]

    return results


def save_selection_results(results: Dict, filepath: str):
    """
    Save selection results to JSON file.

    Args:
        results: Selection results dictionary
        filepath: Output file path
    """
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"Saved selection results to {filepath}")
