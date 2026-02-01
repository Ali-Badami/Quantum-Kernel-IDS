"""
Tests for sensor selection module.
"""

import pytest
import numpy as np

from src.sensor_selection import (
    SensorSelector,
    variance_baseline,
    correlation_baseline,
    random_baseline,
)


class TestSensorSelector:
    """Tests for the SensorSelector class."""

    def test_init(self):
        """Test selector initialization."""
        selector = SensorSelector(regularization=1e-5)
        assert selector.regularization == 1e-5

    def test_log_det(self):
        """Test log-determinant computation."""
        selector = SensorSelector()

        # Identity matrix should have log-det of 0
        K = np.eye(5)
        result = selector._log_det(K, [0, 1, 2])
        assert np.isclose(result, 0, atol=1e-5)

    def test_marginal_gain(self):
        """Test marginal gain computation."""
        selector = SensorSelector()

        # Create a simple correlation matrix
        K = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])

        # Adding first sensor should give positive gain
        gain = selector._marginal_gain(K, [], 0)
        assert gain > 0

    def test_select_greedy(self):
        """Test greedy selection."""
        selector = SensorSelector()

        K = np.eye(5) + 0.1 * np.random.randn(5, 5)
        K = (K + K.T) / 2  # Make symmetric
        np.fill_diagonal(K, 1.0)

        result = selector.select_greedy(K, k=3)

        assert len(result["selected_indices"]) == 3
        assert result["k"] == 3
        assert "objective_value" in result

    def test_select_lazy_greedy(self):
        """Test lazy greedy selection."""
        selector = SensorSelector()

        K = np.eye(5) + 0.1 * np.random.randn(5, 5)
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)

        result = selector.select_lazy_greedy(K, k=3)

        assert len(result["selected_indices"]) == 3
        assert result["algorithm"] == "lazy_greedy"

    def test_greedy_vs_lazy_greedy(self):
        """Test that greedy and lazy greedy give same results."""
        selector = SensorSelector()

        np.random.seed(42)
        K = np.eye(10) + 0.1 * np.random.randn(10, 10)
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)

        result_greedy = selector.select_greedy(K, k=5)
        result_lazy = selector.select_lazy_greedy(K, k=5)

        # Should select same sensors
        assert set(result_greedy["selected_indices"]) == set(result_lazy["selected_indices"])

    def test_with_feature_names(self):
        """Test selection with feature names."""
        selector = SensorSelector()

        K = np.eye(3)
        names = ["sensor_a", "sensor_b", "sensor_c"]

        result = selector.select_greedy(K, k=2, feature_names=names)

        assert "selected_names" in result
        assert len(result["selected_names"]) == 2


class TestBaselines:
    """Tests for baseline selection methods."""

    def test_variance_baseline(self):
        """Test variance-based selection."""
        X = np.array([
            [1, 10, 100],
            [2, 20, 200],
            [3, 30, 300],
        ])

        # Third column has highest variance
        selected = variance_baseline(X, k=1)
        assert 2 in selected

    def test_correlation_baseline(self):
        """Test correlation-based selection."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        selected = correlation_baseline(X, k=3)
        assert len(selected) == 3

    def test_random_baseline(self):
        """Test random selection."""
        selected = random_baseline(n_sensors=10, k=3, seed=42)

        assert len(selected) == 3
        assert len(set(selected)) == 3  # No duplicates

    def test_random_reproducibility(self):
        """Test random baseline reproducibility."""
        selected1 = random_baseline(10, 3, seed=42)
        selected2 = random_baseline(10, 3, seed=42)

        assert selected1 == selected2
