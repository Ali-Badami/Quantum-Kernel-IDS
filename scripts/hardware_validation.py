#!/usr/bin/env python3
"""
Validate quantum kernel on IBM Quantum hardware.

This script runs a subset of the kernel computation on real quantum hardware
and compares results with simulation.

Usage:
    export IBM_QUANTUM_TOKEN="your_token"
    python scripts/hardware_validation.py --backend ibm_fez --samples 20
"""

import argparse
import logging
import os
import sys
import json
import time
from pathlib import Path
from getpass import getpass

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run quantum kernel on IBM Quantum hardware"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="checkpoints/",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ibm_fez",
        help="IBM Quantum backend name",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=15,
        help="Number of qubits",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots per circuit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hardware/",
        help="Output directory",
    )
    return parser.parse_args()


def get_ibm_token():
    """Get IBM Quantum token from environment or prompt."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        token = getpass("Enter IBM Quantum API token: ")
    return token


def connect_ibm_quantum(token):
    """Connect to IBM Quantum service."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        logger.info("Connected using saved credentials")
    except Exception:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=token,
            overwrite=True
        )
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        logger.info("Saved credentials and connected")

    return service


def create_kernel_circuits(X, feature_map):
    """Create fidelity circuits for kernel computation."""
    from qiskit import QuantumCircuit

    n_samples = len(X)
    circuits = []
    circuit_info = []

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Create U(x_i)^dagger U(x_j) circuit
            qc1 = feature_map.assign_parameters(X[i])
            qc2 = feature_map.assign_parameters(X[j])

            circuit = qc2.compose(qc1.inverse())
            circuit.measure_all()

            circuits.append(circuit)
            circuit_info.append((i, j))

    return circuits, circuit_info


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data)
    X_normal = np.load(data_dir / "X_normal.npy")

    # Subsample and reduce dimensions
    n_samples = min(args.samples, len(X_normal))
    n_qubits = args.qubits

    # Use PCA if needed
    if X_normal.shape[1] > n_qubits:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler

        pca = PCA(n_components=n_qubits)
        X_pca = pca.fit_transform(X_normal[:n_samples])
        variance_explained = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA: {X_normal.shape[1]} -> {n_qubits} features, "
                   f"variance: {variance_explained:.2%}")

        scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
        X = scaler.fit_transform(X_pca)
    else:
        X = X_normal[:n_samples]

    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} qubits")

    # Create feature map
    from qiskit.circuit.library import ZZFeatureMap

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=1,
        entanglement="linear",
    )

    # Create circuits
    logger.info("Creating kernel circuits...")
    circuits, circuit_info = create_kernel_circuits(X, feature_map)
    logger.info(f"Created {len(circuits)} circuits")

    # Compute simulator kernel first
    logger.info("Computing simulator kernel...")
    from src.quantum_kernel import QuantumKernelComputer

    kernel_sim = QuantumKernelComputer(n_qubits=n_qubits, reps=1, entanglement="linear")
    K_sim = kernel_sim.compute_matrix(X)
    np.save(output_dir / "K_simulator.npy", K_sim)

    # Connect to IBM Quantum
    logger.info("Connecting to IBM Quantum...")
    token = get_ibm_token()
    service = connect_ibm_quantum(token)

    # Get backend
    backend = service.backend(args.backend)
    logger.info(f"Using backend: {backend.name}")

    # Transpile circuits
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info("Transpiling circuits...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = pm.run(circuits)

    # Run on hardware
    from qiskit_ibm_runtime import SamplerV2

    logger.info(f"Submitting {len(transpiled)} circuits to {backend.name}...")
    logger.info("This may take several minutes...")

    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=args.shots)

    logger.info(f"Job ID: {job.job_id()}")
    logger.info("Waiting for results...")

    result = job.result()

    # Extract kernel values
    K_hw = np.zeros((n_samples, n_samples))

    for idx, (i, j) in enumerate(circuit_info):
        counts = result[idx].data.meas.get_counts()
        all_zeros = "0" * n_qubits
        prob_zero = counts.get(all_zeros, 0) / args.shots
        K_hw[i, j] = prob_zero
        K_hw[j, i] = prob_zero

    np.save(output_dir / "K_hardware.npy", K_hw)

    # Compare kernels
    logger.info("Comparing simulator vs hardware...")

    correlation = np.corrcoef(K_sim.flatten(), K_hw.flatten())[0, 1]
    mae = np.mean(np.abs(K_sim - K_hw))
    rmse = np.sqrt(np.mean((K_sim - K_hw)**2))

    comparison = {
        "correlation": float(correlation),
        "mae": float(mae),
        "rmse": float(rmse),
        "sim_diagonal_mean": float(np.mean(np.diag(K_sim))),
        "hw_diagonal_mean": float(np.mean(np.diag(K_hw))),
        "hw_diagonal_std": float(np.std(np.diag(K_hw))),
    }

    print("\nHardware Validation Results:")
    print("-" * 40)
    print(f"Correlation: {correlation:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Hardware diagonal mean: {comparison['hw_diagonal_mean']:.4f}")

    if correlation > 0.99:
        print("\nAssessment: EXCELLENT - Hardware closely matches simulation")
    elif correlation > 0.95:
        print("\nAssessment: GOOD - Minor deviations from simulation")
    else:
        print("\nAssessment: CAUTION - Significant noise impact")

    # Save results
    results = {
        "job_id": job.job_id(),
        "backend": args.backend,
        "n_samples": n_samples,
        "n_qubits": n_qubits,
        "shots": args.shots,
        "comparison": comparison,
    }

    with open(output_dir / "hardware_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
