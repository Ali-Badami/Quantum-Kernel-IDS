#!/usr/bin/env python3
"""
Select sensors using quantum kernel and baseline methods.

Usage:
    python scripts/select_sensors.py --kernel checkpoints/kernel_matrix.npy --k 25
    python scripts/select_sensors.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensor_selection import (
    SensorSelector,
    compare_methods,
    save_selection_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select sensors using quantum kernel"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help="Path to kernel matrix (.npy file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to original data for baseline methods",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Path to feature names file",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="Number of sensors to select",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/selection_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Regularization parameter",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        k_values = config["selection"]["k_values"]
        regularization = config["selection"]["regularization"]
        output = Path(config["output"]["results_dir"]) / "selection_results.json"
    else:
        k_values = args.k
        regularization = args.regularization
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Load kernel matrix
    logger.info(f"Loading kernel matrix from {args.kernel}")
    K = np.load(args.kernel)
    logger.info(f"Kernel shape: {K.shape}")

    # Load feature names if provided
    feature_names = None
    if args.features:
        with open(args.features) as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(feature_names)} feature names")

    # Load original data for baselines if provided
    X = None
    if args.data:
        X = np.load(args.data)
        logger.info(f"Loaded original data: {X.shape}")

    # Initialize selector
    selector = SensorSelector(regularization=regularization)

    if X is not None:
        # Compare all methods
        logger.info("Comparing quantum and baseline selection methods")
        results = compare_methods(
            K_quantum=K,
            X=X,
            k_values=k_values,
            feature_names=feature_names,
            regularization=regularization,
        )
    else:
        # Just quantum selection
        results = {"quantum": {}}
        for k in k_values:
            logger.info(f"Selecting k={k} sensors")
            results["quantum"][k] = selector.select_lazy_greedy(K, k, feature_names)

    # Print summary
    print("\nSelection Results:")
    print("-" * 60)

    for method in results:
        print(f"\n{method.upper()}:")
        for k in sorted(results[method].keys()):
            obj = results[method][k]["objective_value"]
            if feature_names and "selected_names" in results[method][k]:
                names = results[method][k]["selected_names"][:5]
                print(f"  k={k}: obj={obj:.4f}, top sensors: {names}")
            else:
                indices = results[method][k]["selected_indices"][:5]
                print(f"  k={k}: obj={obj:.4f}, indices: {indices}")

    # Save results
    save_selection_results(results, str(output))
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
