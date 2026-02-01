"""
Quantum kernel computation using Qiskit.

Implements fidelity-based quantum kernels using ZZFeatureMap encoding
for sensor data. Supports both simulation and IBM Quantum hardware execution.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import json
import time

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Install with: pip install qiskit qiskit-aer")


class QuantumKernelComputer:
    """
    Computes quantum kernel matrices using fidelity-based approach.

    The quantum kernel K(x, x') measures the squared overlap between
    quantum states encoding data points x and x':

        K(x, x') = |<0|U†(x)U(x')|0>|²

    where U(x) is the ZZFeatureMap encoding circuit.

    Attributes:
        n_qubits: Number of qubits (must match data dimension)
        reps: Number of feature map repetitions
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        entanglement: str = "linear",
    ):
        """
        Initialize the quantum kernel computer.

        Args:
            n_qubits: Number of qubits for the feature map
            reps: Number of repetitions of the feature map layer
            entanglement: Entanglement topology - 'linear', 'circular', or 'full'
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required. Install with: pip install qiskit qiskit-aer")

        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement

        # Create feature map
        self._feature_map = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )

        # Initialize simulator
        self._simulator = AerSimulator(method="statevector")

        logger.info(f"Initialized quantum kernel: {n_qubits} qubits, "
                   f"reps={reps}, entanglement={entanglement}")

    def _create_kernel_circuit(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> QuantumCircuit:
        """
        Create the fidelity estimation circuit for two data points.

        The circuit applies U(x1)† U(x2) and measures probability of |0...0>.

        Args:
            x1: First data point
            x2: Second data point

        Returns:
            Quantum circuit for kernel evaluation
        """
        # Bind parameters to feature maps
        circuit1 = self._feature_map.assign_parameters(x1)
        circuit2 = self._feature_map.assign_parameters(x2)

        # Create fidelity circuit: U(x1)† U(x2)
        circuit = circuit2.compose(circuit1.inverse())

        return circuit

    def compute_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute a single kernel matrix element.

        Args:
            x1: First data point of shape (n_qubits,)
            x2: Second data point of shape (n_qubits,)

        Returns:
            Kernel value K(x1, x2)
        """
        circuit = self._create_kernel_circuit(x1, x2)

        # Save statevector
        circuit.save_statevector()

        # Run simulation
        job = self._simulator.run(circuit)
        result = job.result()
        statevector = result.get_statevector()

        # Kernel value is probability of measuring |0...0>
        kernel_value = np.abs(statevector[0]) ** 2

        return float(kernel_value)

    def compute_matrix(
        self,
        X: np.ndarray,
        batch_size: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 100,
    ) -> np.ndarray:
        """
        Compute the full kernel matrix for a dataset.

        Args:
            X: Data array of shape (n_samples, n_qubits)
            batch_size: Number of elements per progress update
            checkpoint_dir: Directory to save intermediate results
            checkpoint_freq: Save checkpoint every N elements

        Returns:
            Kernel matrix of shape (n_samples, n_samples)
        """
        n_samples = len(X)

        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Data has {X.shape[1]} features but kernel expects {self.n_qubits}")

        logger.info(f"Computing {n_samples}x{n_samples} kernel matrix...")
        start_time = time.time()

        # Initialize kernel matrix
        K = np.zeros((n_samples, n_samples))

        # Compute upper triangle (matrix is symmetric)
        total_elements = n_samples * (n_samples + 1) // 2
        computed = 0

        with tqdm(total=total_elements, desc="Kernel computation") as pbar:
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if i == j:
                        # Diagonal elements are always 1 for normalized states
                        K[i, j] = 1.0
                    else:
                        K[i, j] = self.compute_element(X[i], X[j])
                        K[j, i] = K[i, j]  # Symmetric

                    computed += 1
                    pbar.update(1)

                    # Checkpoint
                    if checkpoint_dir and computed % checkpoint_freq == 0:
                        self._save_checkpoint(K, checkpoint_dir, computed)

        elapsed = time.time() - start_time
        logger.info(f"Kernel computation complete in {elapsed:.1f} seconds")

        return K

    def _save_checkpoint(self, K: np.ndarray, checkpoint_dir: str, n_computed: int):
        """Save intermediate kernel matrix to checkpoint file."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "kernel_checkpoint.npy", K)
        with open(path / "checkpoint_info.json", "w") as f:
            json.dump({"n_computed": n_computed}, f)

    def validate_kernel(self, K: np.ndarray) -> Dict[str, Any]:
        """
        Validate kernel matrix properties.

        Checks for symmetry, positive semi-definiteness, and proper diagonal.

        Args:
            K: Kernel matrix to validate

        Returns:
            Dictionary with validation results
        """
        n = K.shape[0]

        # Check symmetry
        symmetry_error = np.max(np.abs(K - K.T))
        is_symmetric = symmetry_error < 1e-10

        # Check diagonal (should be 1.0 for normalized states)
        diagonal = np.diag(K)
        diag_mean = np.mean(diagonal)
        diag_std = np.std(diagonal)

        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(K)
        min_eigenvalue = np.min(eigenvalues)
        is_psd = min_eigenvalue >= -1e-10

        # Compute rank
        rank = np.sum(eigenvalues > 1e-10)

        results = {
            "n_samples": n,
            "is_symmetric": bool(is_symmetric),
            "symmetry_error": float(symmetry_error),
            "is_psd": bool(is_psd),
            "min_eigenvalue": float(min_eigenvalue),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "diagonal_mean": float(diag_mean),
            "diagonal_std": float(diag_std),
            "rank": int(rank),
            "off_diagonal_mean": float(np.mean(K[np.triu_indices(n, k=1)])),
        }

        if is_symmetric and is_psd:
            logger.info("Kernel validation PASSED")
        else:
            logger.warning(f"Kernel validation issues: symmetric={is_symmetric}, psd={is_psd}")

        return results


def compute_classical_kernel(
    X: np.ndarray,
    kernel_type: str = "rbf",
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Compute classical kernel matrix for comparison.

    Args:
        X: Data array of shape (n_samples, n_features)
        kernel_type: Type of kernel ('rbf', 'linear', 'polynomial')
        gamma: RBF kernel bandwidth (default: 1/n_features)

    Returns:
        Kernel matrix of shape (n_samples, n_samples)
    """
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    if kernel_type == "rbf":
        return rbf_kernel(X, gamma=gamma)
    elif kernel_type == "linear":
        return linear_kernel(X)
    elif kernel_type == "polynomial":
        return polynomial_kernel(X, degree=3)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def kernel_to_feature_correlation(
    K: np.ndarray,
    X_original: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """
    Transform sample kernel to feature correlation matrix.

    The kernel K is computed between samples, but we need feature-to-feature
    correlation for sensor selection. This computes a weighted covariance
    in the RKHS.

    Args:
        K: Sample kernel matrix of shape (n_samples, n_samples)
        X_original: Original feature data of shape (n_total, n_features)
        sample_indices: Indices of samples used for kernel

    Returns:
        Feature correlation matrix of shape (n_features, n_features)
    """
    X_subset = X_original[sample_indices]
    n_features = X_subset.shape[1]

    # Center the data
    X_centered = X_subset - X_subset.mean(axis=0)

    # Normalize kernel trace
    K_normalized = K / np.trace(K)

    # Weighted covariance: C = X^T K X
    C = X_centered.T @ K_normalized @ X_centered

    # Convert to correlation
    std = np.sqrt(np.diag(C))
    std[std < 1e-10] = 1e-10  # Prevent division by zero
    corr = C / np.outer(std, std)

    # Ensure symmetry and bounds
    corr = (corr + corr.T) / 2
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    return corr
