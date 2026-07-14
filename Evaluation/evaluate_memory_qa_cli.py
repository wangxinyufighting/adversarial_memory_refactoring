"""Evaluate Qwen (or another OpenAI-compatible model) over fixed memories."""

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from case_graph.baseline import AnswerEquivalenceJudge
from case_graph.llm import OpenAIChatClient
from case_graph.retriever import MemoryStore, retriever_config_from_mapping

from .memory_qa import (
    LongMemEvalMemoryAnswerAgent,
    MemoryQuestion,
    discover_memory_case_ids,
    evaluate_memory_question,
    failed_result,
    load_graph_questions,
    load_longmemeval_questions,
    merge_question_metadata,
    summarize_memory_qa,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenAI-compatible answer model on LongMemEval questions "
            "using per-case compressed memories and the training Contriever strategy."
        )
    )
    parser.add_argument("--memory-dir", required=True, help="Directory containing <case_id>.json.")
    parser.add_argument(
        "--dataset",
        default="data/longmemeval/longmemeval_s_cleaned.json",
        help="LongMemEval JSON used to map each memory filename to target QA metadata.",
    )
    parser.add_argument(
        "--graphs",
        help=(
            "Optional test CaseGraph directory/file. When set, graph membership defines "
            "the denominator and cases with missing memory count as failures."
        ),
    )
    parser.add_argument("--coverage-dir", help="Defaults to the sibling coverage_states directory.")
    parser.add_argument(
        "--require-certified-memory",
        action="store_true",
        help="Do not answer cases whose coverage state is absent or not marked done.",
    )
    parser.add_argument("--output", help="Defaults to <memory-dir>/../memory_qa_results.json.")
    parser.add_argument("--case-id", action="append", default=[], help="Evaluate only this case id; repeatable.")
    parser.add_argument("--start-index", type=int, default=1, help="One-based index into selected cases.")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse case results already present in --output.")
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--training-config", default="configs/online_grpo.yaml")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-points", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--retriever-type")
    parser.add_argument("--retriever-model-name")
    parser.add_argument("--retriever-embedding-model")
    parser.add_argument("--retriever-retrieval-mode")
    parser.add_argument("--retriever-device")
    parser.add_argument("--retriever-cache-dir")
    parser.add_argument("--retriever-max-length", type=int)
    retriever_group = parser.add_mutually_exclusive_group()
    retriever_group.add_argument(
        "--retriever-require-model",
        dest="retriever_require_model",
        action="store_true",
        default=None,
    )
    retriever_group.add_argument(
        "--allow-retriever-fallback",
        dest="retriever_require_model",
        action="store_false",
    )

    parser.add_argument(
        "--answer-api-base",
        default=os.environ.get("ANSWER_API_BASE", "http://localhost:8003/v1"),
    )
    parser.add_argument(
        "--answer-model",
        default=os.environ.get("ANSWER_MODEL") or os.environ.get("ATTACKER_MODEL"),
    )
    parser.add_argument(
        "--answer-api-key",
        default=os.environ.get("ANSWER_API_KEY") or os.environ.get("ATTACKER_API_KEY") or "dummy-key",
    )
    parser.add_argument("--answer-timeout", type=int, default=180)
    parser.add_argument("--answer-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--answer-thinking",
        choices=("disabled", "enabled", "server_default"),
        default="disabled",
        help="Qwen3 defaults to disabled thinking for deterministic JSON answers.",
    )

    parser.add_argument("--judge-mode", choices=("llm", "string"), default="llm")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--judge-max-output-tokens", type=int, default=200)
    parser.add_argument(
        "--judge-thinking",
        choices=("disabled", "enabled", "server_default"),
        default="server_default",
    )
    parser.add_argument("--skip-endpoint-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    if args.start_index < 1:
        raise SystemExit("--start-index must be >= 1.")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be >= 1.")
    if args.save_every < 1:
        raise SystemExit("--save-every must be >= 1.")
    if not args.answer_model:
        raise SystemExit(
            "Set ANSWER_MODEL or pass --answer-model. It must exactly match an id from "
            "the answer server's /v1/models response."
        )

    memory_dir = Path(args.memory_dir)
    output_path = (
        Path(args.output)
        if args.output
        else memory_dir.parent / "memory_qa_results.json"
    )
    coverage_dir = (
        Path(args.coverage_dir)
        if args.coverage_dir
        else memory_dir.parent / "coverage_states"
    )

    dataset_questions = load_longmemeval_questions(args.dataset)
    graph_questions = load_graph_questions(args.graphs) if args.graphs else {}
    questions = merge_question_metadata(dataset_questions, graph_questions)
    memory_case_ids = discover_memory_case_ids(memory_dir)
    if graph_questions:
        selected_ids = sorted(graph_questions)
    else:
        selected_ids = sorted(memory_case_ids)

    requested_ids = _flatten_case_ids(args.case_id)
    if requested_ids:
        requested = set(requested_ids)
        selected_ids = [case_id for case_id in selected_ids if case_id in requested]
        absent = sorted(requested - set(selected_ids))
        if absent:
            raise SystemExit(f"Requested case ids were not selected: {', '.join(absent)}")
    selected_ids = selected_ids[args.start_index - 1 :]
    if args.max_cases is not None:
        selected_ids = selected_ids[: args.max_cases]
    if not selected_ids:
        raise SystemExit("No evaluation cases were selected.")

    unmatched_memory_ids = sorted(set(memory_case_ids) - set(dataset_questions))
    missing_question_ids = sorted(set(selected_ids) - set(questions))
    if missing_question_ids:
        raise SystemExit(
            "No target question/answer was found for selected case ids: "
            + ", ".join(missing_question_ids[:20])
        )

    training_config = _load_training_config(args.training_config)
    top_k = int(_setting(args.top_k, training_config, "top_k", 8))
    top_k_points = int(_setting(args.top_k_points, training_config, "top_k_points", 32))
    min_score = float(_setting(args.min_score, training_config, "min_score", 0.0))
    retriever_config = _retriever_config(args, training_config, top_k_points)

    answer_client = OpenAIChatClient(
        model=args.answer_model,
        api_key=args.answer_api_key,
        base_url=args.answer_api_base.rstrip("/"),
        timeout=args.answer_timeout,
        chat_template_kwargs=_thinking_kwargs(args.answer_thinking),
    )
    if not args.skip_endpoint_preflight:
        _preflight_endpoint(answer_client, "answer")

    judge_client = None
    if args.judge_mode == "llm":
        judge_client = _build_judge_client(args, answer_client)
        if not args.skip_endpoint_preflight and judge_client is not answer_client:
            _preflight_endpoint(judge_client, "judge")
    answer_agent = LongMemEvalMemoryAnswerAgent(
        client=answer_client,
        max_output_tokens=args.answer_max_output_tokens,
    )
    judge = AnswerEquivalenceJudge(
        client=judge_client,
        max_output_tokens=args.judge_max_output_tokens,
        use_llm=args.judge_mode == "llm",
    )

    base_payload = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "graphs": str(args.graphs or ""),
        "memory_dir": str(memory_dir),
        "coverage_dir": str(coverage_dir),
        "unmatched_memory_case_ids": unmatched_memory_ids,
        "config": {
            "answer_api_base": answer_client.base_url,
            "answer_model": answer_client.model,
            "answer_thinking": args.answer_thinking,
            "judge_mode": args.judge_mode,
            "judge_api_base": judge_client.base_url if judge_client is not None else "",
            "judge_model": judge_client.model if judge_client is not None else "",
            "retriever": retriever_config,
            "top_k": top_k,
            "top_k_points": top_k_points,
            "min_score": min_score,
            "require_certified_memory": args.require_certified_memory,
        },
    }
    prior_results = _load_prior_results(output_path) if args.resume else []
    results_by_id = {
        str(item.get("case_id")): item
        for item in prior_results
        if str(item.get("case_id")) in set(selected_ids)
    }
    logger.info(
        "Evaluating %d cases: answer_model=%s retriever=%s model=%s device=%s top_k=%d",
        len(selected_ids),
        answer_client.model,
        retriever_config.get("type"),
        retriever_config.get("model_name"),
        retriever_config.get("device") or "auto",
        top_k,
    )

    completed_now = 0
    for index, case_id in enumerate(selected_ids, start=1):
        if case_id in results_by_id:
            logger.info("[%d/%d] %s resumed", index, len(selected_ids), case_id)
            continue
        question = questions[case_id]
        memory_path = memory_dir / f"{case_id}.json"
        coverage_state = _load_json_if_object(coverage_dir / f"{case_id}.json")
        memory_store: Optional[MemoryStore] = None
        if not memory_path.exists():
            result = failed_result(
                question,
                status="missing_memory",
                reason=f"No memory file found at {memory_path}.",
                memory_path=str(memory_path),
                coverage_state=coverage_state,
            )
        else:
            try:
                memory_store = MemoryStore.load(memory_path)
            except Exception as exc:
                result = failed_result(
                    question,
                    status="memory_load_error",
                    reason=str(exc),
                    memory_path=str(memory_path),
                    coverage_state=coverage_state,
                )
            else:
                certification = str(coverage_state.get("completion_status") or "")
                if args.require_certified_memory and certification != "done":
                    result = failed_result(
                        question,
                        status="incomplete_memory",
                        reason=(
                            "Coverage certification is required but completion_status "
                            f"is {certification or 'missing'}."
                        ),
                        memory_path=str(memory_path),
                        memory_store=memory_store,
                        coverage_state=coverage_state,
                    )
                else:
                    try:
                        result = evaluate_memory_question(
                            question,
                            memory_store,
                            answer_agent,
                            judge,
                            retriever_config=retriever_config,
                            top_k=top_k,
                            min_score=min_score,
                            memory_path=str(memory_path),
                            coverage_state=coverage_state,
                        )
                    except Exception as exc:
                        logger.exception("Case %s evaluation failed", case_id)
                        result = failed_result(
                            question,
                            status="evaluation_error",
                            reason=str(exc),
                            memory_path=str(memory_path),
                            memory_store=memory_store,
                            coverage_state=coverage_state,
                        )

        results_by_id[case_id] = result
        completed_now += 1
        ordered_results = [results_by_id[item] for item in selected_ids if item in results_by_id]
        running = summarize_memory_qa(ordered_results)
        logger.info(
            "[%d/%d] %s status=%s correct=%s running_accuracy=%.4f source_recall@%d=%.4f",
            index,
            len(selected_ids),
            case_id,
            result.get("status"),
            result.get("correct"),
            running["accuracy"],
            top_k,
            float((result.get("retrieval_metrics") or {}).get("source_recall") or 0.0),
        )
        if completed_now % args.save_every == 0:
            _write_output(output_path, base_payload, selected_ids, results_by_id)

    payload = _write_output(output_path, base_payload, selected_ids, results_by_id)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Detailed results: {output_path}")


def _build_judge_client(
    args: argparse.Namespace,
    answer_client: OpenAIChatClient,
) -> OpenAIChatClient:
    if not args.judge_api_base and not args.judge_model and not args.judge_api_key:
        return answer_client
    return OpenAIChatClient(
        model=args.judge_model or answer_client.model,
        api_key=args.judge_api_key or answer_client.api_key,
        base_url=(args.judge_api_base or answer_client.base_url).rstrip("/"),
        timeout=args.judge_timeout,
        chat_template_kwargs=_thinking_kwargs(args.judge_thinking),
    )


def _retriever_config(
    args: argparse.Namespace,
    training_config: Dict[str, Any],
    top_k_points: int,
) -> Dict[str, Any]:
    mapping = {"retriever": dict(training_config.get("retriever", {}) or {})}
    overrides = {
        "retriever_type": args.retriever_type,
        "retriever_model_name": args.retriever_model_name,
        "retriever_embedding_model": args.retriever_embedding_model,
        "retriever_retrieval_mode": args.retriever_retrieval_mode,
        "retriever_top_k_points": top_k_points,
        "retriever_device": args.retriever_device,
        "retriever_cache_dir": args.retriever_cache_dir,
        "retriever_require_model": args.retriever_require_model,
        "retriever_max_length": args.retriever_max_length,
    }
    mapping.update({key: value for key, value in overrides.items() if value is not None})
    return retriever_config_from_mapping(mapping)


def _load_training_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Training config does not exist; using evaluator defaults: %s", path)
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return dict(payload)


def _setting(cli_value: Any, training_config: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    value = training_config.get(key, default)
    return default if value is None else value


def _thinking_kwargs(mode: str) -> Optional[Dict[str, Any]]:
    if mode == "server_default":
        return None
    return {"enable_thinking": mode == "enabled"}


def _preflight_endpoint(client: OpenAIChatClient, label: str) -> None:
    request = urllib.request.Request(
        url=f"{client.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {client.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(client.timeout, 30)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"{label.capitalize()} endpoint preflight failed with HTTP {exc.code}: {body}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label.capitalize()} endpoint preflight failed: {exc}") from exc

    model_ids = [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if client.model not in model_ids:
        raise SystemExit(
            f"Configured {label} model {client.model!r} is not served by {client.base_url}. "
            f"Available model ids: {model_ids}. Use the exact id or restart vLLM with "
            "--served-model-name."
        )
    logger.info("%s endpoint ready: model=%s", label.capitalize(), client.model)


def _flatten_case_ids(values: List[str]) -> List[str]:
    result = []
    for value in values:
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def _load_json_if_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load JSON %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_prior_results(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json_if_object(path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    logger.info("Loaded %d prior case results from %s", len(results), path)
    return [item for item in results if isinstance(item, dict)]


def _write_output(
    output_path: Path,
    base_payload: Dict[str, Any],
    selected_ids: List[str],
    results_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    results = [results_by_id[case_id] for case_id in selected_ids if case_id in results_by_id]
    payload = {
        **base_payload,
        "selected_case_ids": selected_ids,
        "summary": summarize_memory_qa(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    main()
