#!/usr/bin/env python3
"""
Run complete experiment pipeline for a dataset.

This script runs preprocessing, kernel computation, sensor selection,
and evaluation in sequence.

Usage:
    python scripts/run_experiment.py --config configs/experiment.yaml --dataset swat
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_dataset
from src.quantum_kernel import QuantumKernelComputer, kernel_to_feature_correlation
from src.sensor_selection import compare_methods, save_selection_results
from src.evaluation import (
    Evaluator,
    run_significance_analysis,
    save_evaluation_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete experiment pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset name from config",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing if files exist",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip kernel computation if file exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = args.dataset or config["dataset"]["name"]
    logger.info(f"Running experiment for dataset: {dataset}")

    # Setup directories
    checkpoint_dir = Path(config["output"]["checkpoint_dir"]) / dataset
    results_dir = Path(config["output"]["results_dir"]) / dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Step 1: Preprocessing
    preproc_file = checkpoint_dir / "X_normal.npy"
    if args.skip_preprocessing and preproc_file.exists():
        logger.info("Skipping preprocessing (files exist)")
    else:
        logger.info("=" * 50)
        logger.info("STEP 1: Preprocessing")
        logger.info("=" * 50)

        preprocess_dataset(
            data_dir=config["dataset"]["data_dir"],
            output_dir=str(checkpoint_dir),
            dataset=dataset,
            n_samples=config["preprocessing"]["n_samples"],
            stride=config["preprocessing"]["stride"],
            pca_components=config["preprocessing"]["pca_components"],
        )

    # Load preprocessed data
    X_normal = np.load(checkpoint_dir / "X_normal.npy")
    X_attack = np.load(checkpoint_dir / "X_attack.npy")

    # Load feature names
    feature_names = None
    feature_file = checkpoint_dir / "feature_list.txt"
    if feature_file.exists():
        with open(feature_file) as f:
            feature_names = [line.strip() for line in f.readlines()]

    logger.info(f"Data loaded: normal={X_normal.shape}, attack={X_attack.shape}")

    # Step 2: Quantum Kernel Computation
    kernel_file = checkpoint_dir / "kernel_matrix.npy"
    if args.skip_kernel and kernel_file.exists():
        logger.info("Skipping kernel computation (file exists)")
        K = np.load(kernel_file)
    else:
        logger.info("=" * 50)
        logger.info("STEP 2: Quantum Kernel Computation")
        logger.info("=" * 50)

        kernel_computer = QuantumKernelComputer(
            n_qubits=config["quantum"]["n_qubits"],
            reps=config["quantum"]["reps"],
            entanglement=config["quantum"]["entanglement"],
        )

        # Use normal data for kernel
        K = kernel_computer.compute_matrix(
            X_normal,
            checkpoint_dir=str(checkpoint_dir),
        )

        np.save(kernel_file, K)

        # Validate
        validation = kernel_computer.validate_kernel(K)
        logger.info(f"Kernel validation: symmetric={validation['is_symmetric']}, "
                   f"psd={validation['is_psd']}")

    # Step 3: Sensor Selection
    logger.info("=" * 50)
    logger.info("STEP 3: Sensor Selection")
    logger.info("=" * 50)

    # Convert to feature correlation if needed
    if K.shape[0] != X_normal.shape[1]:
        logger.info("Converting sample kernel to feature correlation")
        sample_indices = np.load(checkpoint_dir / "sample_indices_normal.npy")
        K_features = kernel_to_feature_correlation(K, X_normal, sample_indices)
    else:
        K_features = K

    selection_results = compare_methods(
        K_quantum=K_features,
        X=X_normal,
        k_values=config["selection"]["k_values"],
        feature_names=feature_names,
        regularization=config["selection"]["regularization"],
    )

    selection_file = results_dir / "selection_results.json"
    save_selection_results(selection_results, str(selection_file))

    # Step 4: Evaluation
    logger.info("=" * 50)
    logger.info("STEP 4: Evaluation")
    logger.info("=" * 50)

    evaluator = Evaluator(
        seeds=config["evaluation"]["seeds"],
        test_ratio=config["evaluation"]["test_ratio"],
    )

    eval_results = evaluator.compare_methods(
        X_normal=X_normal,
        X_attack=X_attack,
        selection_results=selection_results,
        classifier=config["evaluation"]["primary_classifier"],
    )

    # Significance tests
    significance = run_significance_analysis(
        eval_results,
        reference_method="quantum",
        metric="f1",
        alpha=config["evaluation"]["alpha"],
    )

    # Save results
    full_results = {
        "dataset": dataset,
        "evaluation": eval_results,
        "significance": significance,
        "config": config,
    }

    eval_file = results_dir / "evaluation_results.json"
    save_evaluation_results(full_results, str(eval_file))

    # Summary
    total_time = time.time() - total_start
    logger.info("=" * 50)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Results saved to: {results_dir}")

    # Print best results
    print("\nBest Results (F1 Score):")
    print("-" * 40)
    for method in eval_results:
        best_k = max(eval_results[method].keys(),
                    key=lambda k: eval_results[method][k]["f1_mean"])
        best_f1 = eval_results[method][best_k]["f1_mean"]
        best_std = eval_results[method][best_k]["f1_std"]
        print(f"  {method}: {best_f1:.4f} ± {best_std:.4f} (k={best_k})")


if __name__ == "__main__":
    main()
