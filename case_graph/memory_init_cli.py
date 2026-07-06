import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an initial memory store from CaseGraph chunks.")
    parser.add_argument("--graphs", required=True, help="CaseGraph JSON file or directory.")
    parser.add_argument("--output", required=True, help="Output memory store JSON path.")
    parser.add_argument("--max-graphs", type=int, default=-1, help="Limit graphs used; -1 means all.")
    args = parser.parse_args()

    memories = []
    for graph_path in _graph_paths(Path(args.graphs), args.max_graphs):
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        case_id = str(graph.get("case_id") or graph_path.stem.replace(".case_graph", ""))
        for chunk in graph.get("chunks", []):
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            content = str(chunk.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            memories.append(
                {
                    "memory_id": f"{case_id}:{chunk_id}",
                    "content": content,
                    "linked_questions": [],
                    "metadata": {
                        "case_id": case_id,
                        "source_chunk_id": chunk_id,
                        "timestamp": chunk.get("timestamp", ""),
                    },
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"memories": memories}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(memories)} memories to {output_path}")


def _graph_paths(path: Path, max_graphs: int) -> list[Path]:
    paths = sorted(path.glob("*.case_graph.json")) if path.is_dir() else [path]
    return paths if max_graphs < 0 else paths[:max_graphs]


if __name__ == "__main__":
    main()
