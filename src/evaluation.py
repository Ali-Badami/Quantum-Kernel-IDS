"""
Evaluation framework for sensor selection quality.

Provides multi-seed cross-validation, statistical significance testing,
and comprehensive metric computation for anomaly detection performance.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Multi-seed evaluation for sensor selection methods.

    Performs repeated train/test splits with different random seeds,
    computes performance metrics, and runs statistical significance tests.

    Attributes:
        seeds: List of random seeds for reproducibility
        test_ratio: Fraction of data used for testing
        classifiers: Dictionary of classifier configurations
    """

    DEFAULT_CLASSIFIERS = {
        "ocsvm": {"class": OneClassSVM, "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"}},
        "iforest": {"class": IsolationForest, "params": {"n_estimators": 100, "contamination": 0.1, "random_state": 42}},
        "rf": {"class": RandomForestClassifier, "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
        "svm": {"class": SVC, "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42}},
    }

    def __init__(
        self,
        seeds: List[int] = None,
        test_ratio: float = 0.2,
        classifiers: Dict[str, Dict] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            seeds: Random seeds for repeated evaluation
            test_ratio: Fraction of data for testing
            classifiers: Classifier configurations (name -> config dict)
        """
        self.seeds = seeds or [42, 123, 456, 789, 1000]
        self.test_ratio = test_ratio
        self.classifiers = classifiers or self.DEFAULT_CLASSIFIERS

    def _create_split(
        self,
        X_normal: np.ndarray,
        X_attack: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split with given seed.

        Training set contains only normal data (for one-class methods)
        or both classes (for supervised methods).
        Test set always contains both classes.

        Args:
            X_normal: Normal operation samples
            X_attack: Attack samples
            seed: Random seed for split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        rng = np.random.RandomState(seed)

        # Shuffle indices
        idx_normal = rng.permutation(len(X_normal))
        idx_attack = rng.permutation(len(X_attack))

        # Split sizes
        n_train_normal = int(len(X_normal) * (1 - self.test_ratio))
        n_train_attack = int(len(X_attack) * (1 - self.test_ratio))

        # Training data (both classes for supervised, normal only for one-class)
        X_train_normal = X_normal[idx_normal[:n_train_normal]]
        X_train_attack = X_attack[idx_attack[:n_train_attack]]

        # Test data
        X_test_normal = X_normal[idx_normal[n_train_normal:]]
        X_test_attack = X_attack[idx_attack[n_train_attack:]]

        # Combine for supervised learning
        X_train = np.vstack([X_train_normal, X_train_attack])
        y_train = np.array([0] * len(X_train_normal) + [1] * len(X_train_attack))

        X_test = np.vstack([X_test_normal, X_test_attack])
        y_test = np.array([0] * len(X_test_normal) + [1] * len(X_test_attack))

        return X_train, X_test, y_train, y_test

    def _evaluate_classifier(
        self,
        clf_name: str,
        clf_config: Dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train and evaluate a single classifier.

        Args:
            clf_name: Classifier name
            clf_config: Classifier configuration
            X_train, X_test, y_train, y_test: Train/test data

        Returns:
            Dictionary of performance metrics
        """
        clf_class = clf_config["class"]
        clf_params = clf_config["params"].copy()

        clf = clf_class(**clf_params)

        # One-class methods train only on normal data
        if clf_name in ["ocsvm", "iforest"]:
            X_train_normal = X_train[y_train == 0]
            clf.fit(X_train_normal)
            y_pred_raw = clf.predict(X_test)
            # Convert: -1 (anomaly) -> 1 (attack), 1 (normal) -> 0
            y_pred = np.where(y_pred_raw == -1, 1, 0)
        else:
            # Supervised classifiers
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        # AUC requires probability scores (not all classifiers support this)
        try:
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test)
                if clf_name in ["ocsvm", "iforest"]:
                    y_score = -y_score  # Negate for anomaly detection
            else:
                y_score = y_pred

            metrics["auc"] = roc_auc_score(y_test, y_score)
        except Exception:
            metrics["auc"] = 0.0

        return metrics

    def evaluate_selection(
        self,
        X_normal: np.ndarray,
        X_attack: np.ndarray,
        selected_indices: List[int],
        classifier: str = "rf",
    ) -> Dict[str, Any]:
        """
        Evaluate sensor selection with multiple random seeds.

        Args:
            X_normal: Normal operation data
            X_attack: Attack data
            selected_indices: Indices of selected sensors
            classifier: Which classifier to use

        Returns:
            Dictionary with mean/std metrics and per-seed results
        """
        # Select only the chosen sensors
        X_normal_sel = X_normal[:, selected_indices]
        X_attack_sel = X_attack[:, selected_indices]

        clf_config = self.classifiers[classifier]
        all_metrics = {metric: [] for metric in ["accuracy", "precision", "recall", "f1", "auc"]}

        for seed in self.seeds:
            X_train, X_test, y_train, y_test = self._create_split(
                X_normal_sel, X_attack_sel, seed
            )

            metrics = self._evaluate_classifier(
                classifier, clf_config, X_train, X_test, y_train, y_test
            )

            for metric, value in metrics.items():
                all_metrics[metric].append(value)

        # Compute summary statistics
        result = {
            "k": len(selected_indices),
            "classifier": classifier,
            "n_seeds": len(self.seeds),
            "seeds": self.seeds,
        }

        for metric, values in all_metrics.items():
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
            result[f"{metric}_scores"] = [float(v) for v in values]

            # 95% confidence interval
            if len(values) > 1:
                ci = stats.t.interval(
                    0.95,
                    df=len(values) - 1,
                    loc=np.mean(values),
                    scale=stats.sem(values),
                )
                result[f"{metric}_ci_lower"] = float(ci[0])
                result[f"{metric}_ci_upper"] = float(ci[1])

        return result

    def compare_methods(
        self,
        X_normal: np.ndarray,
        X_attack: np.ndarray,
        selection_results: Dict[str, Dict[int, Dict]],
        classifier: str = "rf",
    ) -> Dict[str, Dict[int, Dict]]:
        """
        Compare multiple selection methods across k values.

        Args:
            X_normal: Normal operation data
            X_attack: Attack data
            selection_results: Results from sensor_selection.compare_methods
            classifier: Classifier to use for evaluation

        Returns:
            Nested dict: method -> k -> evaluation results
        """
        eval_results = {}

        for method, k_results in selection_results.items():
            eval_results[method] = {}

            for k, sel_result in k_results.items():
                k_int = int(k)
                logger.info(f"Evaluating {method}, k={k_int}")

                indices = sel_result["selected_indices"]
                eval_results[method][k_int] = self.evaluate_selection(
                    X_normal, X_attack, indices, classifier
                )

        return eval_results


def significance_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical significance tests between two methods.

    Uses paired t-test (parametric) and Wilcoxon signed-rank test (non-parametric).

    Args:
        scores_a: Scores from method A (e.g., quantum)
        scores_b: Scores from method B (e.g., baseline)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_a - scores_b

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(scores_a, scores_b)

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pvalue = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        # All differences are zero
        w_stat, w_pvalue = 0.0, 1.0

    # Effect size (Cohen's d)
    if np.std(diff) > 0:
        cohens_d = np.mean(diff) / np.std(diff)
    else:
        cohens_d = 0.0

    return {
        "mean_diff": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "cohens_d": float(cohens_d),
        "significant_ttest": int(t_pvalue < alpha),
        "significant_wilcoxon": int(w_pvalue < alpha),
        "a_better": int(np.mean(diff) > 0),
    }


def run_significance_analysis(
    eval_results: Dict[str, Dict[int, Dict]],
    reference_method: str = "quantum",
    metric: str = "f1",
    alpha: float = 0.05,
) -> Dict[str, Dict[int, Dict]]:
    """
    Run significance tests comparing reference method to all baselines.

    Args:
        eval_results: Results from Evaluator.compare_methods
        reference_method: Method to compare against
        metric: Metric to use for comparison
        alpha: Significance level

    Returns:
        Nested dict: baseline -> k -> significance test results
    """
    significance = {}

    ref_results = eval_results.get(reference_method, {})

    for method, k_results in eval_results.items():
        if method == reference_method:
            continue

        significance[method] = {}

        for k, results in k_results.items():
            k_int = int(k)
            if k_int not in ref_results:
                continue

            ref_scores = ref_results[k_int][f"{metric}_scores"]
            method_scores = results[f"{metric}_scores"]

            significance[method][k_int] = significance_test(
                ref_scores, method_scores, alpha
            )

    return significance


def save_evaluation_results(
    results: Dict,
    filepath: str,
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        filepath: Output file path
    """
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"Saved evaluation results to {filepath}")


def generate_summary_table(
    eval_results: Dict[str, Dict[int, Dict]],
    significance: Dict[str, Dict[int, Dict]],
    metric: str = "f1",
    k_values: List[int] = None,
) -> str:
    """
    Generate a summary table in Markdown format.

    Args:
        eval_results: Evaluation results
        significance: Significance test results
        metric: Metric to summarize
        k_values: Which k values to include

    Returns:
        Markdown formatted table string
    """
    methods = list(eval_results.keys())
    if k_values is None:
        k_values = sorted(set().union(*[r.keys() for r in eval_results.values()]))

    lines = ["| k | " + " | ".join(methods) + " |"]
    lines.append("|---" + "|---" * len(methods) + "|")

    for k in k_values:
        row = [str(k)]
        for method in methods:
            if k in eval_results.get(method, {}):
                mean = eval_results[method][k][f"{metric}_mean"]
                std = eval_results[method][k][f"{metric}_std"]
                cell = f"{mean:.3f} ± {std:.3f}"

                # Add significance marker
                if method != "quantum" and method in significance:
                    if k in significance[method]:
                        if significance[method][k]["significant_ttest"]:
                            if significance[method][k]["a_better"]:
                                cell += " **"
                            else:
                                cell += " *"
            else:
                cell = "-"
            row.append(cell)

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
