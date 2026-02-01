#!/usr/bin/env python3
"""
Preprocess ICS dataset for quantum kernel computation.

Usage:
    python scripts/preprocess.py --dataset swat --output checkpoints/
    python scripts/preprocess.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ICS dataset for quantum kernel computation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swat",
        choices=["swat", "wadi", "hai"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing raw dataset files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/",
        help="Output directory for preprocessed files",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per class",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Temporal subsampling stride",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=20,
        help="Number of PCA components (0 to skip PCA)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides other args)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        dataset = config["dataset"]["name"]
        data_dir = config["dataset"]["data_dir"]
        output = config["output"]["checkpoint_dir"]
        n_samples = config["preprocessing"]["n_samples"]
        stride = config["preprocessing"]["stride"]
        pca = config["preprocessing"]["pca_components"]
    else:
        dataset = args.dataset
        data_dir = args.data_dir
        output = args.output
        n_samples = args.samples
        stride = args.stride
        pca = args.pca if args.pca > 0 else None

    logger.info(f"Preprocessing {dataset} dataset")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output}")

    # Run preprocessing
    summary = preprocess_dataset(
        data_dir=data_dir,
        output_dir=output,
        dataset=dataset,
        n_samples=n_samples,
        stride=stride,
        pca_components=pca,
    )

    # Print summary
    print("\nPreprocessing Summary:")
    print(f"  Dataset: {summary['dataset']}")
    print(f"  Original features: {summary['n_features_original']}")
    print(f"  Active features: {summary['n_features_active']}")
    print(f"  Removed features: {summary['n_removed']}")
    print(f"  Normal samples: {summary['n_normal_samples']}")
    print(f"  Attack samples: {summary['n_attack_samples']}")
    if pca:
        print(f"  PCA components: {summary['pca_components']}")
    print(f"\nOutput saved to: {output}")


if __name__ == "__main__":
    main()
