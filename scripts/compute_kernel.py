#!/usr/bin/env python3
"""
Compute quantum kernel matrix for preprocessed data.

Usage:
    python scripts/compute_kernel.py --data checkpoints/X_samples.npy --output checkpoints/
    python scripts/compute_kernel.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum_kernel import QuantumKernelComputer, kernel_to_feature_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute quantum kernel matrix"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed data (.npy file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/",
        help="Output directory for kernel matrix",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=20,
        help="Number of qubits",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Feature map repetitions",
    )
    parser.add_argument(
        "--entanglement",
        type=str,
        default="linear",
        choices=["linear", "circular", "full"],
        help="Entanglement pattern",
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

        qubits = config["quantum"]["n_qubits"]
        reps = config["quantum"]["reps"]
        entanglement = config["quantum"]["entanglement"]
        output = config["output"]["checkpoint_dir"]
    else:
        qubits = args.qubits
        reps = args.reps
        entanglement = args.entanglement
        output = args.output

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data}")
    X = np.load(args.data)
    logger.info(f"Data shape: {X.shape}")

    if X.shape[1] != qubits:
        logger.warning(f"Data has {X.shape[1]} features but {qubits} qubits specified. "
                      f"Using {X.shape[1]} qubits.")
        qubits = X.shape[1]

    # Initialize kernel computer
    logger.info(f"Initializing quantum kernel: {qubits} qubits, reps={reps}, "
               f"entanglement={entanglement}")
    kernel_computer = QuantumKernelComputer(
        n_qubits=qubits,
        reps=reps,
        entanglement=entanglement,
    )

    # Compute kernel matrix
    start_time = time.time()
    K = kernel_computer.compute_matrix(X, checkpoint_dir=str(output_path))
    elapsed = time.time() - start_time

    logger.info(f"Kernel computation completed in {elapsed:.1f} seconds")

    # Validate kernel
    validation = kernel_computer.validate_kernel(K)
    print("\nKernel Validation:")
    print(f"  Shape: {K.shape}")
    print(f"  Symmetric: {validation['is_symmetric']}")
    print(f"  Positive semi-definite: {validation['is_psd']}")
    print(f"  Diagonal mean: {validation['diagonal_mean']:.6f}")
    print(f"  Off-diagonal mean: {validation['off_diagonal_mean']:.6f}")
    print(f"  Min eigenvalue: {validation['min_eigenvalue']:.6f}")
    print(f"  Rank: {validation['rank']}")

    # Save kernel matrix
    kernel_path = output_path / "kernel_matrix.npy"
    np.save(kernel_path, K)
    logger.info(f"Saved kernel matrix to {kernel_path}")

    # Save metadata
    import json
    metadata = {
        "n_samples": int(X.shape[0]),
        "n_qubits": qubits,
        "reps": reps,
        "entanglement": entanglement,
        "computation_time_sec": elapsed,
        "validation": validation,
    }
    with open(output_path / "kernel_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nKernel matrix saved to: {kernel_path}")


if __name__ == "__main__":
    main()
