# Dataset Documentation

## Supported Datasets

This repository supports three Industrial Control System (ICS) security datasets.

## SWaT (Secure Water Treatment)

**Source**: [iTrust, Singapore University of Technology and Design](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

### Description
A 6-stage water treatment testbed including:
- P1: Raw water storage
- P2: Chemical dosing
- P3: Ultrafiltration
- P4: Dechlorination (UV treatment)
- P5: Reverse osmosis
- P6: Backwash cleaning

### Features
- **Total sensors**: 51 (25 sensors + 26 actuators)
- **Active after preprocessing**: 42
- **Sample rate**: 1 second

### Feature Types
| Type | Count | Description |
|------|-------|-------------|
| FIT | 6 | Flow indicators |
| LIT | 4 | Level indicators |
| AIT | 9 | Analyzers (pH, conductivity, etc.) |
| PIT | 3 | Pressure indicators |
| DPIT | 1 | Differential pressure |
| MV | 5 | Motorized valves |
| P | 14 | Pumps |

### Dataset Statistics
| Split | Duration | Samples | Description |
|-------|----------|---------|-------------|
| Normal | 7 days | ~605,000 | Normal operation |
| Attack | 4 days | ~449,000 | 41 attack scenarios |

### Zero-Variance Features
The following features have constant values and are removed:
- P202, P301, P401, P404, P502, P601, P603

---

## WADI (Water Distribution)

**Source**: [iTrust, Singapore University of Technology and Design](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

### Description
A water distribution network testbed simulating:
- Consumer water demand
- Distribution pipelines
- Storage tanks
- Pumping stations

### Features
- **Total sensors**: 123 (69 sensors + 54 actuators)
- **Active after preprocessing**: 93
- **Sample rate**: 1 second

### Feature Types
| Type | Count | Description |
|------|-------|-------------|
| AIT | 18 | Analyzers |
| FIT | 5 | Flow indicators |
| PIT | 4 | Pressure indicators |
| LIT | 4 | Level indicators |
| LS | 40 | Level switches |
| MV | 10 | Motorized valves |
| P | 25 | Pumps |

### Dataset Statistics
| Split | Duration | Samples | Description |
|-------|----------|---------|-------------|
| Normal | 14 days | ~784,000 | Normal operation |
| Attack | 2 days | ~173,000 | 15 attack scenarios |

---

## HAI (Hardware-in-the-Loop Augmented ICS)

**Source**: [KISTI, Korea](https://github.com/icsdataset/hai)

### Description
A testbed combining:
- P1: Boiler process
- P2: Turbine process
- P3: Water treatment
- P4: Auxiliary systems

### Features
- **Total sensors**: 86
- **Active after preprocessing**: 68
- **Sample rate**: 1 second

### Dataset Statistics (HAI 22.04)
| Split | Samples | Description |
|-------|---------|-------------|
| Train | Variable | Normal + some attacks |
| Test | Variable | Attack scenarios |

---

## Obtaining Datasets

### SWaT and WADI
1. Visit [iTrust Dataset Request](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
2. Fill out the academic use agreement form
3. Datasets will be emailed after approval (typically 1-2 weeks)

### HAI
1. Visit [HAI GitHub Repository](https://github.com/icsdataset/hai)
2. Download directly from releases
3. No registration required

---

## Data Format

Place downloaded files in the `data/` directory:

```
data/
├── swat/
│   ├── SWaT_Dataset_Normal_v1.csv
│   └── SWaT_Dataset_Attack_v0.csv
├── wadi/
│   ├── WADI_14days_new.csv
│   └── WADI_attackdataLABLE.csv
└── hai/
    ├── train.csv
    └── test.csv
```

The data loader automatically detects file names and formats.

---

## Preprocessing Details

### Step 1: Load Data
- Automatic encoding detection (UTF-8, Latin-1)
- Automatic separator detection (comma, semicolon, tab)
- Header row detection and correction

### Step 2: Handle Missing Values
- Forward fill (ffill)
- Backward fill (bfill)
- Remaining NaN → 0

### Step 3: Remove Zero-Variance Features
- Variance threshold: 10⁻¹⁰
- Known constant features removed

### Step 4: Normalize
- MinMax scaling to [0, 2π]
- Range suitable for quantum gate rotations

### Step 5: PCA (Optional)
- Reduce dimensionality for quantum circuits
- Default: 20 components
- Typically captures >95% variance

### Step 6: Temporal Subsampling
- Stride-based sampling (default: 100)
- Reduces autocorrelation
- Maintains temporal coverage
