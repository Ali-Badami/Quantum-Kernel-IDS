"""
Data loading utilities for ICS security datasets.

Supports loading SWaT, WADI, and HAI datasets with automatic format detection
and column name normalization.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and manages ICS security datasets.

    Handles multiple file formats and naming conventions used across
    different dataset versions.

    Attributes:
        data_dir: Base directory containing dataset files
        dataset: Name of the dataset (swat, wadi, or hai)
    """

    # Known dataset configurations
    DATASET_CONFIGS = {
        "swat": {
            "normal_patterns": ["SWaT_Dataset_Normal*", "normal*", "*Normal*"],
            "attack_patterns": ["SWaT_Dataset_Attack*", "attack*", "*Attack*"],
            "label_column": ["Normal/Attack", "label", "Label", "attack"],
            "timestamp_column": ["Timestamp", "timestamp", "Time", "datetime"],
        },
        "wadi": {
            "normal_patterns": ["WADI_14days*", "*normal*", "WADI.A1*"],
            "attack_patterns": ["WADI_attackdata*", "*attack*", "WADI.A2*"],
            "label_column": ["Attack LABLE", "Attack", "label", "Label"],
            "timestamp_column": ["Date", "Time", "Row", "Timestamp"],
        },
        "hai": {
            "normal_patterns": ["train*", "*normal*", "hai*train*"],
            "attack_patterns": ["test*", "*attack*", "hai*test*"],
            "label_column": ["attack", "Attack", "label", "Label"],
            "timestamp_column": ["timestamp", "time", "Timestamp"],
        },
    }

    def __init__(self, data_dir: str, dataset: str = "swat"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing dataset files
            dataset: Dataset name - one of 'swat', 'wadi', 'hai'
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset.lower()

        if self.dataset not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}. "
                           f"Supported: {list(self.DATASET_CONFIGS.keys())}")

        self.config = self.DATASET_CONFIGS[self.dataset]
        self._normal_df = None
        self._attack_df = None

    def find_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Locate normal and attack data files in the data directory.

        Returns:
            Tuple of (normal_file_path, attack_file_path)
        """
        normal_file = None
        attack_file = None

        # Search for normal data
        for pattern in self.config["normal_patterns"]:
            matches = list(self.data_dir.glob(f"**/{pattern}.csv"))
            if matches:
                normal_file = matches[0]
                break

        # Search for attack data
        for pattern in self.config["attack_patterns"]:
            matches = list(self.data_dir.glob(f"**/{pattern}.csv"))
            if matches:
                attack_file = matches[0]
                break

        if normal_file:
            logger.info(f"Found normal data: {normal_file}")
        else:
            logger.warning("Could not locate normal data file")

        if attack_file:
            logger.info(f"Found attack data: {attack_file}")
        else:
            logger.warning("Could not locate attack data file")

        return normal_file, attack_file

    def _read_csv_flexible(self, filepath: Path) -> pd.DataFrame:
        """
        Read CSV with automatic separator and encoding detection.

        Args:
            filepath: Path to the CSV file

        Returns:
            Loaded DataFrame
        """
        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        separators = [",", ";", "\t"]

        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, sep=sep,
                                    low_memory=False)
                    # Check if we got reasonable columns
                    if len(df.columns) > 1:
                        logger.debug(f"Read {filepath} with encoding={encoding}, sep='{sep}'")
                        return df
                except Exception:
                    continue

        raise ValueError(f"Could not read {filepath} with any encoding/separator combination")

    def _detect_header_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if first row contains column names instead of data.

        Some datasets have the header as the first data row if pandas
        didn't detect it properly.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with corrected headers
        """
        first_col = str(df.columns[0])

        # If column names are numeric strings, header wasn't detected
        if first_col.isdigit() or first_col == "0":
            logger.info("Header not detected, extracting from first row")
            new_header = df.iloc[0].tolist()
            df = df[1:].reset_index(drop=True)
            df.columns = new_header

        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        Find a column matching one of the candidate names.

        Args:
            df: DataFrame to search
            candidates: List of possible column names

        Returns:
            Matched column name or None
        """
        df_cols_lower = {c.lower().strip(): c for c in df.columns}

        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            if candidate_lower in df_cols_lower:
                return df_cols_lower[candidate_lower]

        return None

    def _extract_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract and convert labels to binary format.

        Args:
            df: DataFrame with label column

        Returns:
            Tuple of (DataFrame without label, binary label array)
        """
        label_col = self._find_column(df, self.config["label_column"])

        if label_col is None:
            logger.warning("No label column found, assuming all samples are normal")
            return df, np.zeros(len(df), dtype=int)

        labels = df[label_col].copy()
        df_features = df.drop(columns=[label_col])

        # Convert to binary: 0 = normal, 1 = attack
        if labels.dtype == object:
            labels = labels.str.lower().str.strip()
            binary_labels = np.where(
                labels.isin(["normal", "0", "false", "no"]), 0, 1
            ).astype(int)
        else:
            binary_labels = (labels != 0).astype(int)

        return df_features, binary_labels

    def load_normal(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load the normal operation dataset.

        Returns:
            Tuple of (feature DataFrame, label array)
        """
        if self._normal_df is not None:
            return self._normal_df

        normal_file, _ = self.find_files()
        if normal_file is None:
            raise FileNotFoundError("Could not find normal data file")

        df = self._read_csv_flexible(normal_file)
        df = self._detect_header_row(df)

        # Remove timestamp columns
        ts_col = self._find_column(df, self.config["timestamp_column"])
        if ts_col:
            df = df.drop(columns=[ts_col])

        df, labels = self._extract_labels(df)

        logger.info(f"Loaded normal data: {df.shape[0]} samples, {df.shape[1]} features")
        self._normal_df = (df, labels)

        return df, labels

    def load_attack(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load the attack dataset.

        Returns:
            Tuple of (feature DataFrame, label array)
        """
        if self._attack_df is not None:
            return self._attack_df

        _, attack_file = self.find_files()
        if attack_file is None:
            raise FileNotFoundError("Could not find attack data file")

        df = self._read_csv_flexible(attack_file)
        df = self._detect_header_row(df)

        # Remove timestamp columns
        ts_col = self._find_column(df, self.config["timestamp_column"])
        if ts_col:
            df = df.drop(columns=[ts_col])

        df, labels = self._extract_labels(df)

        logger.info(f"Loaded attack data: {df.shape[0]} samples, {df.shape[1]} features")
        self._attack_df = (df, labels)

        return df, labels

    def load_combined(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and combine normal and attack datasets.

        Returns:
            Tuple of (combined feature DataFrame, combined label array)
        """
        df_normal, y_normal = self.load_normal()
        df_attack, y_attack = self.load_attack()

        # Find common columns
        common_cols = list(set(df_normal.columns) & set(df_attack.columns))
        if len(common_cols) < len(df_normal.columns):
            logger.warning(f"Only {len(common_cols)} common columns found")

        df_normal = df_normal[common_cols]
        df_attack = df_attack[common_cols]

        df_combined = pd.concat([df_normal, df_attack], ignore_index=True)
        y_combined = np.concatenate([y_normal, y_attack])

        logger.info(f"Combined data: {df_combined.shape[0]} samples, "
                   f"normal={np.sum(y_combined==0)}, attack={np.sum(y_combined==1)}")

        return df_combined, y_combined

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names.

        Returns:
            List of feature names
        """
        df, _ = self.load_normal()
        return list(df.columns)


def load_preprocessed(checkpoint_dir: str) -> Dict[str, np.ndarray]:
    """
    Load preprocessed numpy arrays from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing .npy files

    Returns:
        Dictionary mapping filenames to numpy arrays
    """
    checkpoint_path = Path(checkpoint_dir)
    data = {}

    expected_files = [
        "X_normal.npy",
        "X_attack.npy",
        "X_normal_raw.npy",
        "X_attack_raw.npy",
    ]

    for filename in expected_files:
        filepath = checkpoint_path / filename
        if filepath.exists():
            data[filename.replace(".npy", "")] = np.load(filepath)
            logger.info(f"Loaded {filename}: shape {data[filename.replace('.npy', '')].shape}")

    # Load feature list if available
    feature_file = checkpoint_path / "feature_list.txt"
    if feature_file.exists():
        with open(feature_file, "r") as f:
            data["feature_names"] = [line.strip() for line in f.readlines()]

    return data
