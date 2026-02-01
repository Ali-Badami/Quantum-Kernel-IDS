#!/usr/bin/env python3
"""
Evaluate sensor selection with multi-seed cross-validation.

Usage:
    python scripts/evaluate.py --data checkpoints/ --sensors results/selection_results.json
    python scripts/evaluate.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    Evaluator,
    run_significance_analysis,
    save_evaluation_results,
    generate_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sensor selection performance"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        required=True,
        help="Path to selection results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1000],
        help="Random seeds for evaluation",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=["ocsvm", "iforest", "rf", "svm"],
        help="Classifier to use",
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

        seeds = config["evaluation"]["seeds"]
        classifier = config["evaluation"]["primary_classifier"]
        output = Path(config["output"]["results_dir"]) / "evaluation_results.json"
        alpha = config["evaluation"]["alpha"]
    else:
        seeds = args.seeds
        classifier = args.classifier
        output = Path(args.output)
        alpha = 0.05

    output.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data)
    logger.info(f"Loading data from {data_dir}")

    X_normal = np.load(data_dir / "X_normal.npy")
    X_attack = np.load(data_dir / "X_attack.npy")

    logger.info(f"Normal samples: {X_normal.shape}")
    logger.info(f"Attack samples: {X_attack.shape}")

    # Load selection results
    logger.info(f"Loading selection results from {args.sensors}")
    with open(args.sensors) as f:
        selection_results = json.load(f)

    # Initialize evaluator
    evaluator = Evaluator(seeds=seeds)

    # Evaluate all methods
    logger.info(f"Evaluating with {len(seeds)} seeds using {classifier} classifier")
    eval_results = evaluator.compare_methods(
        X_normal=X_normal,
        X_attack=X_attack,
        selection_results=selection_results,
        classifier=classifier,
    )

    # Run significance tests
    logger.info("Running significance tests")
    significance = run_significance_analysis(
        eval_results,
        reference_method="quantum",
        metric="f1",
        alpha=alpha,
    )

    # Combine results
    full_results = {
        "evaluation": eval_results,
        "significance": significance,
        "config": {
            "seeds": seeds,
            "classifier": classifier,
            "alpha": alpha,
            "n_normal": int(X_normal.shape[0]),
            "n_attack": int(X_attack.shape[0]),
            "n_features": int(X_normal.shape[1]),
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDataset: {X_normal.shape[0]} normal, {X_attack.shape[0]} attack samples")
    print(f"Features: {X_normal.shape[1]}")
    print(f"Classifier: {classifier}")
    print(f"Seeds: {seeds}")

    # Print F1 scores table
    print("\nF1 Scores (mean ± std):")
    print("-" * 70)
    table = generate_summary_table(eval_results, significance, metric="f1")
    print(table)

    # Print significance summary
    print("\nStatistical Significance (Quantum vs Baselines):")
    print("-" * 70)
    for baseline in significance:
        sig_count = sum(1 for k in significance[baseline]
                       if significance[baseline][k]["significant_ttest"])
        total = len(significance[baseline])
        print(f"  vs {baseline}: {sig_count}/{total} k-values significant (p<{alpha})")

    # Save results
    save_evaluation_results(full_results, str(output))
    print(f"\nResults saved to: {output}")

    # Also save summary table
    table_path = output.parent / "summary_table.md"
    with open(table_path, "w") as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"Classifier: {classifier}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write("## F1 Scores\n\n")
        f.write(table)
        f.write("\n\n** = Quantum significantly better (p<0.05)\n")
        f.write("* = Baseline significantly better (p<0.05)\n")

    print(f"Summary table saved to: {table_path}")


if __name__ == "__main__":
    main()
