import argparse
import json
from pathlib import Path

from .retriever import FrozenBM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve Top-K old-memory chunks from Mt.")
    parser.add_argument("--memory", required=True, help="Path to memory store JSON.")
    parser.add_argument("--question", required=True, help="Question Q to retrieve for.")
    parser.add_argument("--output", help="Optional path to write retrieval hits as JSON.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = FrozenBM25Retriever.from_path(args.memory)
    hits = retriever.retrieve(args.question, top_k=args.top_k, min_score=args.min_score)
    payload = {
        "question": args.question,
        "top_k": args.top_k,
        "retrieved_memories": [hit.to_dict() for hit in hits],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
