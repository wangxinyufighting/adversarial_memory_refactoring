import argparse
import json

from .grpo_adapter import write_verl_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verl parquet data for memory-refactoring GRPO.")
    parser.add_argument("--states", required=True, help="JSON list of memory refactoring training states.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    args = parser.parse_args()

    with open(args.states, "r", encoding="utf-8") as handle:
        states = json.load(handle)
    write_verl_parquet(states, args.output)


if __name__ == "__main__":
    main()
