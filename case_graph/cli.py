import argparse
import json
from pathlib import Path

from .builder import CaseGraphBuilder
from .cache import load_extraction_cache, save_extraction_cache
from .llm import LLMExtractor
from .longmemeval import entry_to_session_chunks
from .progress import ProgressBar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-case entity-relation graphs from LongMemEval JSON.")
    parser.add_argument("--input", required=True, help="Path to LongMemEval JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for graph JSON outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of cases to process.")
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Persistent extraction cache JSON. Defaults to <output-dir>/extraction_cache.json.",
    )
    parser.add_argument(
        "--include-raw-chunks",
        action="store_true",
        help="Write raw session text into graph JSON. Enabled by default for verification evidence.",
    )
    parser.add_argument(
        "--omit-raw-chunks",
        action="store_true",
        help="Omit raw session text from graph JSON.",
    )
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="Include assistant turns in session chunks. Default matches UnifiedMem and keeps user turns only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "extraction_cache.json"

    entries = json.loads(input_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        entries = entries[: args.limit]

    extractor = LLMExtractor()
    extraction_cache = load_extraction_cache(cache_path)
    progress_bar = ProgressBar(label="case chunks")
    manifest = []
    total_cases = len(entries)
    for case_index, entry in enumerate(entries, start=1):
        case_id = entry.get("case_id") or entry.get("question_id") or entry.get("id") or "unknown_case"
        chunks = entry_to_session_chunks(entry, include_assistant=args.include_assistant)
        builder = CaseGraphBuilder(
            extractor=extractor,
            extraction_cache=extraction_cache,
            progress_callback=lambda current, total, chunk, idx=case_index: progress_bar.update(
                current,
                total,
                suffix=f"case {idx}/{total_cases} {chunk.chunk_id}",
            ),
        )
        graph = builder.build(case_id=case_id, chunks=chunks)
        if entry.get("answer") is not None:
            graph.ensure_target_answer(
                question=entry.get("question", ""),
                answer=entry.get("answer"),
                source_ids=entry.get("answer_session_ids", []),
            )
        out_path = output_dir / f"{case_id}.case_graph.json"
        include_chunk_content = args.include_raw_chunks or not args.omit_raw_chunks
        out_path.write_text(
            json.dumps(
                graph.to_dict(include_chunk_content=include_chunk_content),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.append({"case_id": case_id, "path": str(out_path), "num_chunks": len(chunks)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_extraction_cache(cache_path, extraction_cache)
    print(f"Wrote {len(manifest)} case graphs to {output_dir}")
    print(f"Extraction cache: {cache_path}")


if __name__ == "__main__":
    main()
