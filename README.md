# Quantum Kernel Sensor Selection for ICS Anomaly Detection

Implementation of quantum kernel-based sensor selection for Industrial Control System (ICS) anomaly detection. This repository accompanies our IEEE Access paper on using quantum machine learning for critical infrastructure security.

## Overview

We propose using quantum kernels computed via parameterized quantum circuits to perform D-optimal sensor selection for ICS anomaly detection. The key insight is that quantum kernels can capture complex correlations in high-dimensional sensor data that classical methods may miss.

**Key Features:**
- Quantum kernel computation using ZZFeatureMap encoding
- Greedy submodular sensor selection maximizing log-determinant
- Support for three ICS benchmark datasets (SWaT, WADI, HAI)
- Both simulation and IBM Quantum hardware execution
- Comprehensive evaluation with statistical significance testing

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for simulation)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-kernel-ids.git
cd quantum-kernel-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
qiskit>=1.3.0
qiskit-ibm-runtime>=0.34.0
qiskit-aer>=0.15.0
qiskit-machine-learning>=0.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
```

## Datasets

This work uses three publicly available ICS security datasets:

| Dataset | Source | Features | Description |
|---------|--------|----------|-------------|
| SWaT | [iTrust, SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | 51 | Secure Water Treatment testbed |
| WADI | [iTrust, SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | 123 | Water Distribution network |
| HAI | [KISTI](https://github.com/icsdataset/hai) | 86 | Hardware-in-the-loop Augmented ICS |

**Note:** Datasets must be requested from the respective sources. Place downloaded files in `data/` directory.

## Quick Start

### 1. Preprocess Data

```bash
python scripts/preprocess.py --dataset swat --output checkpoints/
```

### 2. Compute Quantum Kernel

```bash
python scripts/compute_kernel.py \
    --data checkpoints/X_samples.npy \
    --qubits 20 \
    --output checkpoints/kernel_matrix.npy
```

### 3. Run Sensor Selection

```bash
python scripts/select_sensors.py \
    --kernel checkpoints/kernel_matrix.npy \
    --k 25 \
    --output results/selected_sensors.json
```

### 4. Evaluate Detection Performance

```bash
python scripts/evaluate.py \
    --data checkpoints/ \
    --sensors results/selected_sensors.json \
    --seeds 5 \
    --output results/evaluation.json
```

## Repository Structure

```
quantum-kernel-ids/
├── src/                    # Core source code
│   ├── data_loader.py      # Dataset loading utilities
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── quantum_kernel.py   # Quantum kernel computation
│   ├── sensor_selection.py # Greedy selection algorithm
│   └── evaluation.py       # Evaluation metrics
├── notebooks/              # Jupyter notebooks for experiments
│   ├── 01_preprocessing.ipynb
│   ├── 02_quantum_kernel.ipynb
│   ├── 03_sensor_selection.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_hardware_validation.ipynb
├── configs/                # Configuration files
│   └── experiment.yaml
├── scripts/                # Command-line scripts
│   ├── preprocess.py
│   ├── compute_kernel.py
│   ├── select_sensors.py
│   └── evaluate.py
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── results/                # Output directory
```

## Methodology

### Quantum Kernel

We use the ZZFeatureMap to encode classical sensor data into quantum states:

```
U(x) = ∏[H ⊗ RZ(x_i) ⊗ exp(-i x_i x_j ZZ)]
```

The quantum kernel measures state overlap:

```
K(x, x') = |⟨0|U†(x)U(x')|0⟩|²
```

### Sensor Selection

Given the kernel matrix K, we select sensors by greedy maximization of the log-determinant:

```
S* = argmax_{|S|=k} log det(K_S + λI)
```

This is a submodular function, guaranteeing a (1-1/e) approximation ratio.

## Configuration

Edit `configs/experiment.yaml` to customize experiments:

```yaml
quantum:
  n_qubits: 20
  feature_map: ZZFeatureMap
  entanglement: linear
  reps: 1

selection:
  k_values: [5, 10, 15, 20, 25, 30]
  regularization: 1.0e-6

evaluation:
  seeds: [42, 123, 456, 789, 1000]
  test_ratio: 0.2
  classifiers: [ocsvm, iforest, rf, svm]
```

## Results

### Main Results (F1 Score at k=25)

| Dataset | Quantum | Variance | Correlation | Random |
|---------|---------|----------|-------------|--------|
| SWaT    | **0.955** | 0.949  | 0.946       | 0.948  |
| WADI    | **0.976** | 0.941  | 0.954       | 0.971  |
| HAI     | **0.917** | 0.905  | 0.919       | 0.910  |

### Hardware Validation

Validated on IBM Quantum ibm_fez (156 qubits):
- Simulator-hardware correlation: >0.999
- Sensor selection agreement: 100%
- Job ID: `d5sm2k4bmr9c739m4io0`

## Reproducing Results

To reproduce all experiments from the paper:

```bash
# Run full pipeline for all datasets
./scripts/run_all.sh

# Or run individually:
python scripts/run_experiment.py --config configs/experiment.yaml --dataset swat
python scripts/run_experiment.py --config configs/experiment.yaml --dataset wadi
python scripts/run_experiment.py --config configs/experiment.yaml --dataset hai
```

## Hardware Execution

To run on IBM Quantum hardware:

```bash
# Set your IBM Quantum API token
export IBM_QUANTUM_TOKEN="your_token_here"

# Run hardware validation
python scripts/hardware_validation.py \
    --backend ibm_fez \
    --samples 20 \
    --shots 1024
```

**Note:** Hardware execution requires an IBM Quantum account. Free tier provides 10 minutes/month.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{quantumkernelids2026,
  title={Quantum Kernel-Based Sensor Selection for Industrial Control System Anomaly Detection},
  author={},
  journal={IEEE Access},
  year={2026},
  doi={}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- iTrust, Singapore University of Technology and Design for SWaT and WADI datasets
- KISTI, Korea for HAI dataset
- IBM Quantum for hardware access
