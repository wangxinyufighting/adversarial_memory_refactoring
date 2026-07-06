import argparse
import json
from pathlib import Path

from .refactoring import SimilarityActionRouter
from .retriever import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute similarity and choose Add/Merge without mutating memory."
    )
    parser.add_argument("--memory", required=True, help="Current memory JSON path.")
    parser.add_argument("--question", required=True, help="Current attack question Q.")
    parser.add_argument("--tau", required=True, type=float, help="Similarity threshold.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of memories to retrieve.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    store = MemoryStore.load(args.memory)
    decision = SimilarityActionRouter(tau=args.tau, top_k=args.top_k).choose_action(
        question=args.question,
        memory_store=store,
    )
    payload = decision.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
