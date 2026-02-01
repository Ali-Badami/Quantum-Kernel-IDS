"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.preprocessing import Preprocessor


class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_init(self):
        """Test preprocessor initialization."""
        prep = Preprocessor(pca_components=10)
        assert prep.pca_components == 10
        assert prep.quantum_range == (0, 2 * np.pi)

    def test_handle_missing_values(self):
        """Test missing value handling."""
        prep = Preprocessor()

        df = pd.DataFrame({
            "a": [1, np.nan, 3, 4, 5],
            "b": [np.nan, 2, 3, 4, 5],
        })

        result = prep._handle_missing_values(df)
        assert not result.isna().any().any()

    def test_remove_constant_features(self):
        """Test zero-variance feature removal."""
        prep = Preprocessor()

        df = pd.DataFrame({
            "varying": [1, 2, 3, 4, 5],
            "constant": [1, 1, 1, 1, 1],
        })

        result = prep._remove_constant_features(df)
        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_normalize_for_quantum(self):
        """Test quantum normalization."""
        prep = Preprocessor()

        X = np.array([[0, 100], [50, 200], [100, 300]])
        result = prep._normalize_for_quantum(X)

        assert result.min() >= 0
        assert result.max() <= 2 * np.pi

    def test_fit_transform(self):
        """Test complete preprocessing pipeline."""
        prep = Preprocessor(pca_components=2)

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 1.0, 1.0, 1.0],  # constant
        })

        X, _ = prep.fit_transform(df)

        # Should have 2 PCA components
        assert X.shape[1] == 2
        # Should be normalized
        assert X.min() >= 0
        assert X.max() <= 2 * np.pi

    def test_temporal_subsample(self):
        """Test temporal subsampling."""
        prep = Preprocessor()

        X = np.random.randn(1000, 10)
        X_sub, _, indices = prep.temporal_subsample(X, stride=100)

        assert len(X_sub) == 10
        assert len(indices) == 10
        assert indices[1] - indices[0] == 100

    def test_save_state(self):
        """Test state saving."""
        prep = Preprocessor()

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })

        prep.fit_transform(df)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            prep.save_state(f.name)
            assert Path(f.name).exists()


class TestTemporalSubsampling:
    """Tests for temporal subsampling functionality."""

    def test_max_samples(self):
        """Test max_samples parameter."""
        prep = Preprocessor()

        X = np.random.randn(1000, 10)
        X_sub, _, _ = prep.temporal_subsample(X, stride=10, max_samples=50)

        assert len(X_sub) == 50

    def test_with_labels(self):
        """Test subsampling with labels."""
        prep = Preprocessor()

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X_sub, y_sub, indices = prep.temporal_subsample(X, y, stride=10)

        assert len(X_sub) == len(y_sub)
        assert np.array_equal(y_sub, y[indices])
