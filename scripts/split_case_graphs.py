#!/usr/bin/env python3
"""Split case graphs into train/val/test sets."""

import argparse
import random
import shutil
from pathlib import Path


def split_graphs(
    graphs_dir: str,
    output_base: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Split graphs into train/val/test sets."""
    graphs_path = Path(graphs_dir)
    output_path = Path(output_base)

    # Get all graph files
    graph_files = sorted(graphs_path.glob("*.case_graph.json"))
    total = len(graph_files)

    if total == 0:
        print(f"No graph files found in {graphs_dir}")
        return

    print(f"Found {total} graph files in {graphs_dir}")
    print(f"Train ratio: {train_ratio}")
    print(f"Val ratio: {val_ratio}")
    print(f"Test ratio: {1 - train_ratio - val_ratio:.2f}")
    print(f"Seed: {seed}")
    print()

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(graph_files)

    # Calculate split points
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    print(f"Split: Train={train_count}, Val={val_count}, Test={test_count}")
    print()

    # Create output directories
    train_dir = output_path / "case_graphs_train"
    val_dir = output_path / "case_graphs_val"
    test_dir = output_path / "case_graphs_test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Split and copy files
    train_files = graph_files[:train_count]
    val_files = graph_files[train_count:train_count + val_count]
    test_files = graph_files[train_count + val_count:]

    print("Train set:")
    for f in train_files:
        shutil.copy(f, train_dir / f.name)
        print(f"  {f.name}")

    print("\nValidation set:")
    for f in val_files:
        shutil.copy(f, val_dir / f.name)
        print(f"  {f.name}")

    print("\nTest set:")
    for f in test_files:
        shutil.copy(f, test_dir / f.name)
        print(f"  {f.name}")

    print("\nSplit complete!")
    print(f"Train: {train_dir}/ ({train_count} files)")
    print(f"Val: {val_dir}/ ({val_count} files)")
    print(f"Test: {test_dir}/ ({test_count} files)")


def main():
    parser = argparse.ArgumentParser(description="Split case graphs into train/val/test")
    parser.add_argument(
        "graphs_dir",
        help="Directory containing case graph JSON files"
    )
    parser.add_argument(
        "--output-base",
        default="outputs",
        help="Base directory for output (default: outputs)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training set ratio (default: 0.6)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    split_graphs(
        graphs_dir=args.graphs_dir,
        output_base=args.output_base,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
