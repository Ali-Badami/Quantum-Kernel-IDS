"""
Data preprocessing pipeline for quantum kernel computation.

Handles missing values, zero-variance feature removal, normalization,
and temporal subsampling for ICS time-series data.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessing pipeline for ICS sensor data.

    Applies a sequence of transformations to prepare data for quantum
    kernel computation:
    1. Handle missing values (forward/backward fill)
    2. Remove zero-variance features
    3. Normalize to [0, 2*pi] for quantum gate rotations
    4. Optional PCA for dimensionality reduction
    5. Temporal subsampling to reduce autocorrelation

    Attributes:
        quantum_range: Normalization range for quantum encoding, default (0, 2*pi)
        remove_zero_variance: Whether to remove constant features
        pca_components: Number of PCA components (None to skip PCA)
    """

    # Features known to be constant in SWaT dataset
    SWAT_ZERO_VARIANCE = ["P202", "P301", "P401", "P404", "P502", "P601", "P603"]

    def __init__(
        self,
        quantum_range: Tuple[float, float] = (0, 2 * np.pi),
        remove_zero_variance: bool = True,
        pca_components: Optional[int] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            quantum_range: Target range for normalization (min, max)
            remove_zero_variance: Remove features with zero variance
            pca_components: Number of PCA components, None to skip
        """
        self.quantum_range = quantum_range
        self.remove_zero_variance = remove_zero_variance
        self.pca_components = pca_components

        self._scaler = None
        self._pca = None
        self._removed_features = []
        self._feature_names = []
        self._normalization_params = {}

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward and backward fill.

        For time-series data, this preserves temporal continuity better
        than mean imputation.

        Args:
            df: Input DataFrame with possible NaN values

        Returns:
            DataFrame with missing values filled
        """
        initial_nan = df.isna().sum().sum()

        if initial_nan == 0:
            return df

        # Forward fill first, then backward fill for any remaining
        df = df.ffill().bfill()

        # If still NaN (entire columns are empty), fill with 0
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"{remaining_nan} NaN values remain after fill, setting to 0")
            df = df.fillna(0)

        logger.info(f"Filled {initial_nan} missing values")
        return df

    def _remove_constant_features(
        self, df: pd.DataFrame, threshold: float = 1e-10
    ) -> pd.DataFrame:
        """
        Remove features with near-zero variance.

        Args:
            df: Input DataFrame
            threshold: Variance threshold below which features are removed

        Returns:
            DataFrame with constant features removed
        """
        variances = df.var()
        low_var_cols = variances[variances < threshold].index.tolist()

        # Also check for known zero-variance features
        for col in self.SWAT_ZERO_VARIANCE:
            if col in df.columns and col not in low_var_cols:
                low_var_cols.append(col)

        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} zero-variance features: {low_var_cols}")
            self._removed_features = low_var_cols
            df = df.drop(columns=low_var_cols)

        return df

    def _normalize_for_quantum(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize data to quantum-compatible range [0, 2*pi].

        Uses min-max scaling per feature to map values to the target range.
        This ensures each feature spans the full rotation range of quantum gates.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Normalized array in range [0, 2*pi]
        """
        min_val, max_val = self.quantum_range

        # Fit scaler on training data
        if self._scaler is None:
            self._scaler = MinMaxScaler(feature_range=(min_val, max_val))
            X_normalized = self._scaler.fit_transform(X)

            # Store parameters for reproducibility
            self._normalization_params = {
                "data_min": self._scaler.data_min_.tolist(),
                "data_max": self._scaler.data_max_.tolist(),
                "range_min": min_val,
                "range_max": max_val,
            }
        else:
            X_normalized = self._scaler.transform(X)

        return X_normalized

    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Reduced array of shape (n_samples, n_components)
        """
        if self.pca_components is None:
            return X

        if self._pca is None:
            self._pca = PCA(n_components=self.pca_components)
            X_reduced = self._pca.fit_transform(X)

            variance_explained = np.sum(self._pca.explained_variance_ratio_)
            logger.info(f"PCA: {X.shape[1]} -> {self.pca_components} features, "
                       f"variance explained: {variance_explained:.2%}")
        else:
            X_reduced = self._pca.transform(X)

        return X_reduced

    def temporal_subsample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        stride: int = 100,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Subsample time-series data with fixed stride.

        Reduces autocorrelation and dataset size while maintaining
        temporal coverage.

        Args:
            X: Input array of shape (n_samples, n_features)
            y: Optional label array
            stride: Take every nth sample
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (subsampled X, subsampled y, indices used)
        """
        n_samples = len(X)
        indices = np.arange(0, n_samples, stride)

        if max_samples and len(indices) > max_samples:
            # Randomly select subset of strided indices
            rng = np.random.RandomState(42)
            indices = rng.choice(indices, size=max_samples, replace=False)
            indices.sort()

        X_sub = X[indices]
        y_sub = y[indices] if y is not None else None

        logger.info(f"Subsampled: {n_samples} -> {len(indices)} samples (stride={stride})")

        return X_sub, y_sub, indices

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessor and transform the data.

        Args:
            df: Input DataFrame
            y: Optional label array

        Returns:
            Tuple of (transformed X, y)
        """
        # Store feature names
        self._feature_names = list(df.columns)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove zero-variance features
        if self.remove_zero_variance:
            df = self._remove_constant_features(df)

        # Update feature names after removal
        self._feature_names = list(df.columns)

        # Convert to numpy
        X = df.values.astype(np.float64)

        # Apply PCA before normalization (on standardized data)
        if self.pca_components:
            std_scaler = StandardScaler()
            X_std = std_scaler.fit_transform(X)
            X_pca = self._apply_pca(X_std)
            X = X_pca

        # Normalize for quantum encoding
        X_quantum = self._normalize_for_quantum(X)

        return X_quantum, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.

        Args:
            df: Input DataFrame

        Returns:
            Transformed array
        """
        if self._scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = self._handle_missing_values(df)

        # Remove same features as training
        for col in self._removed_features:
            if col in df.columns:
                df = df.drop(columns=[col])

        X = df.values.astype(np.float64)

        if self.pca_components and self._pca:
            std_scaler = StandardScaler()
            X_std = std_scaler.fit_transform(X)
            X = self._pca.transform(X_std)

        return self._scaler.transform(X)

    def get_feature_names(self) -> List[str]:
        """Return list of active feature names after preprocessing."""
        return self._feature_names

    def get_removed_features(self) -> List[str]:
        """Return list of features removed during preprocessing."""
        return self._removed_features

    def save_state(self, filepath: str):
        """
        Save preprocessor state for reproducibility.

        Args:
            filepath: Path to save JSON state file
        """
        state = {
            "quantum_range": self.quantum_range,
            "pca_components": self.pca_components,
            "feature_names": self._feature_names,
            "removed_features": self._removed_features,
            "normalization_params": self._normalization_params,
        }

        if self._pca:
            state["pca_variance_explained"] = self._pca.explained_variance_ratio_.tolist()

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved preprocessor state to {filepath}")


def preprocess_dataset(
    data_dir: str,
    output_dir: str,
    dataset: str = "swat",
    n_samples: int = 1000,
    stride: int = 100,
    pca_components: Optional[int] = 20,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for a dataset.

    Args:
        data_dir: Directory containing raw dataset files
        output_dir: Directory to save preprocessed files
        dataset: Dataset name (swat, wadi, hai)
        n_samples: Number of samples to extract per class
        stride: Temporal subsampling stride
        pca_components: Number of PCA components

    Returns:
        Dictionary with preprocessing summary
    """
    from .data_loader import DataLoader

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    loader = DataLoader(data_dir, dataset)
    df_normal, y_normal = loader.load_normal()
    df_attack, y_attack = loader.load_attack()

    # Initialize preprocessor
    preprocessor = Preprocessor(pca_components=pca_components)

    # Fit on normal data
    X_normal, _ = preprocessor.fit_transform(df_normal, y_normal)

    # Transform attack data
    X_attack = preprocessor.transform(df_attack)

    # Subsample
    X_normal_sub, _, idx_normal = preprocessor.temporal_subsample(
        X_normal, stride=stride, max_samples=n_samples
    )
    X_attack_sub, _, idx_attack = preprocessor.temporal_subsample(
        X_attack, stride=stride, max_samples=n_samples
    )

    # Save outputs
    np.save(output_path / "X_normal.npy", X_normal_sub)
    np.save(output_path / "X_attack.npy", X_attack_sub)
    np.save(output_path / "sample_indices_normal.npy", idx_normal)
    np.save(output_path / "sample_indices_attack.npy", idx_attack)

    # Save feature names
    with open(output_path / "feature_list.txt", "w") as f:
        for name in preprocessor.get_feature_names():
            f.write(f"{name}\n")

    # Save preprocessor state
    preprocessor.save_state(str(output_path / "preprocessor_state.json"))

    summary = {
        "dataset": dataset,
        "n_features_original": len(df_normal.columns),
        "n_features_active": len(preprocessor.get_feature_names()),
        "n_removed": len(preprocessor.get_removed_features()),
        "n_normal_samples": len(X_normal_sub),
        "n_attack_samples": len(X_attack_sub),
        "pca_components": pca_components,
        "stride": stride,
    }

    with open(output_path / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Preprocessing complete. Saved to {output_path}")
    return summary
