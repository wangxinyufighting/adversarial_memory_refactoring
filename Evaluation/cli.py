"""CLI for controlled memory evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from baselines import (
    MemoryDirectoryBaseline,
    OracleSourceBaseline,
    RawSessionBaseline,
    UnifiedMemBaseline,
)
from baselines.unifiedmem import DEFAULT_UNIFIEDMEM_REPO
from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.defense import RetrievedMemoryAnswerAgent
from case_graph.llm import OpenAIChatClient

from .core import evaluate_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate memory methods with shared retrieval and answer backbone."
    )
    parser.add_argument("--graphs", required=True, help="CaseGraph file or directory.")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=["raw_session", "unifiedmem", "oracle_source"],
        default=[],
        help="Built-in baseline to evaluate. Can be repeated.",
    )
    parser.add_argument(
        "--memory-dir",
        action="append",
        default=[],
        help="Evaluate a per-case memory directory as NAME=DIR or DIR.",
    )
    parser.add_argument("--unifiedmem-repo", default=DEFAULT_UNIFIEDMEM_REPO)
    parser.add_argument(
        "--unifiedmem-expansion-cache",
        action="append",
        default=[],
        help="UnifiedMem expansion cache JSON. Can be repeated.",
    )
    parser.add_argument(
        "--unifiedmem-join-mode",
        default="merge",
        choices=["merge", "merge_raw", "separate", "none"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--answer-max-output-tokens", type=int, default=200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument("--skip-llm-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = _build_baselines(args)
    if not baselines:
        raise SystemExit("Provide at least one --baseline or --memory-dir.")

    client = OpenAIChatClient.from_env()
    answer_agent = RetrievedMemoryAnswerAgent(
        client=client,
        max_output_tokens=args.answer_max_output_tokens,
    )
    judge = AnswerEquivalenceJudge(
        client=client,
        max_output_tokens=args.judge_max_output_tokens,
        use_llm=not args.skip_llm_judge,
    )
    payload = evaluate_suite(
        graphs=args.graphs,
        baselines=baselines,
        answer_agent=answer_agent,
        judge=judge,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"leaderboard": payload["leaderboard"]}, ensure_ascii=False, indent=2))


def _build_baselines(args: argparse.Namespace) -> List[object]:
    baselines: List[object] = []
    for item in args.memory_dir:
        name, path = _parse_named_path(item, default_name="candidate_memory")
        baselines.append(MemoryDirectoryBaseline(memory_dir=path, name=name))

    for name in args.baseline:
        if name == "raw_session":
            baselines.append(RawSessionBaseline())
        elif name == "unifiedmem":
            baselines.append(
                UnifiedMemBaseline(
                    repo_path=args.unifiedmem_repo,
                    expansion_cache_paths=args.unifiedmem_expansion_cache,
                    join_mode=args.unifiedmem_join_mode,
                )
            )
        elif name == "oracle_source":
            baselines.append(OracleSourceBaseline())
    return baselines


def _parse_named_path(value: str, default_name: str) -> tuple[str, str]:
    if "=" not in value:
        return default_name, value
    name, path = value.split("=", 1)
    return name or default_name, path


if __name__ == "__main__":
    main()
