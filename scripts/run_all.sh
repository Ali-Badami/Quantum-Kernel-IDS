#!/bin/bash
# Run experiments for all datasets
# Usage: ./scripts/run_all.sh

set -e

echo "=================================================="
echo "Quantum Kernel Sensor Selection - Full Experiment"
echo "=================================================="

CONFIG="configs/experiment.yaml"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Run for each dataset
for DATASET in swat wadi hai; do
    echo ""
    echo "=================================================="
    echo "Running experiment for: $DATASET"
    echo "=================================================="

    python scripts/run_experiment.py \
        --config "$CONFIG" \
        --dataset "$DATASET"

    echo "Completed: $DATASET"
done

echo ""
echo "=================================================="
echo "All experiments complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - results/swat/"
echo "  - results/wadi/"
echo "  - results/hai/"
