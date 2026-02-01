# Methodology

## Overview

This document describes the methodology for quantum kernel-based sensor selection in Industrial Control System (ICS) anomaly detection.

## Quantum Kernel Computation

### Feature Encoding

We use the ZZFeatureMap to encode classical sensor data into quantum states. For a data point x ∈ ℝⁿ, the quantum state is prepared by:

```
|φ(x)⟩ = U(x)|0⟩^⊗n
```

where U(x) is the ZZFeatureMap circuit:

```
U(x) = ∏ᵣ [∏ᵢ Hᵢ · ∏ᵢ RZ(xᵢ) · ∏ᵢ<ⱼ exp(-i xᵢxⱼ ZᵢZⱼ)]
```

The parameter r controls the circuit depth (repetitions).

### Kernel Evaluation

The quantum kernel measures the similarity between two data points as:

```
K(x, x') = |⟨φ(x)|φ(x')⟩|²
```

This is computed by:
1. Preparing U(x')
2. Applying U(x)†
3. Measuring probability of |0...0⟩

### Implementation

We use Qiskit's statevector simulator for efficient kernel computation. For n samples, we compute the upper triangle of the n×n kernel matrix (symmetric), requiring n(n+1)/2 circuit evaluations.

## Sensor Selection

### D-Optimal Design

We select sensors by maximizing the log-determinant of the kernel submatrix:

```
S* = argmax_{|S|=k} log det(K_S + λI)
```

where:
- K_S is the kernel matrix restricted to selected sensors
- λ is a small regularization term (default: 10⁻⁶)
- k is the number of sensors to select

### Submodularity

The log-determinant objective is submodular, enabling efficient greedy approximation with a (1 - 1/e) ≈ 63.2% approximation guarantee.

### Algorithm

We use lazy greedy selection for efficiency:

1. Initialize: S = ∅, priority queue Q with all sensors
2. For i = 1 to k:
   - Pop sensor with highest upper bound from Q
   - If bound is current, select sensor
   - Otherwise, recompute gain and push back
3. Return S

Lazy evaluation typically reduces complexity from O(kn·n³) to O(kn³).

## Evaluation

### Multi-Seed Cross-Validation

We evaluate using 5 random seeds to assess stability:
- Seeds: [42, 123, 456, 789, 1000]
- Each seed creates a different train/test split
- Report mean ± std of metrics

### Classifiers

We evaluate with multiple classifiers:
- **One-Class SVM**: Trained on normal data only
- **Isolation Forest**: Unsupervised anomaly detection
- **Random Forest**: Supervised classification
- **SVM-RBF**: Supervised with RBF kernel

### Metrics

For each evaluation:
- **Accuracy**: Overall correct classification rate
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Statistical Significance

We compare methods using:
- **Paired t-test**: Parametric comparison
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Cohen's d**: Effect size measure
- **95% Confidence Intervals**: Via t-distribution

## Baselines

We compare against classical selection methods:

1. **Variance**: Select sensors with highest variance
2. **Correlation**: Greedy selection on Pearson correlation matrix
3. **Random**: Randomly select sensors
4. **RBF Kernel**: Greedy selection on classical RBF kernel

## Hardware Validation

For IBM Quantum validation:
- Use smaller sample count (20 per dataset)
- Reduce qubits via PCA (15 qubits)
- Linear entanglement for hardware compatibility
- 1024 shots per circuit
- Compare with simulation results
