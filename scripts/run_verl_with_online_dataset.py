#!/usr/bin/env python3
"""Wrapper script to run verl with OnlineMemoryDataset.

This script monkey-patches verl's dataset loading to use our custom dataset.
"""

import json
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our dataset
from case_graph.online_memory_dataset import OnlineMemoryDataset


def patch_verl_dataset_loader():
    """Monkey-patch verl's get_dataset_class to return our dataset."""
    from verl.utils.dataset import rl_dataset

    original_get_dataset_class = rl_dataset.get_dataset_class

    def custom_get_dataset_class(data_config):
        """Return OnlineMemoryDataset if marker file exists."""
        marker_file = Path("online_dataset_config.json")

        if marker_file.exists():
            print(f"Loading OnlineMemoryDataset from {marker_file}")
            with open(marker_file) as f:
                config = json.load(f)

            # Create a wrapper class that verl can instantiate
            class OnlineDatasetWrapper(OnlineMemoryDataset):
                def __init__(self, *args, **kwargs):
                    # Ignore verl's standard arguments, use our config
                    super().__init__(
                        graph_files=config["graph_files"],
                        config=config["config"],
                        initial_memory_dir=config.get("initial_memory_dir"),
                    )

            return OnlineDatasetWrapper
        else:
            # Fall back to verl's default
            return original_get_dataset_class(data_config)

    # Replace the function
    rl_dataset.get_dataset_class = custom_get_dataset_class
    print("Patched verl dataset loader")


if __name__ == "__main__":
    # Apply the patch
    patch_verl_dataset_loader()

    # Now run verl's main_ppo
    from verl.trainer.main_ppo import main

    print("Starting verl training with OnlineMemoryDataset")
    main()
