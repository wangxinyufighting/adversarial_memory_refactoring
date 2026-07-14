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

from .agents import LongMemEvalAnswerJudge, LongMemEvalMemoryAnswerAgent
from .evaluator import evaluate_memory_question, failed_result
from .loaders import (
    MemoryQuestion,
    case_graph_to_memory_store,
    discover_memory_case_ids,
    load_case_graphs,
    load_longmemeval_questions,
    merge_question_metadata,
)
from .metrics import compute_answer_metrics, compute_retrieval_metrics, memory_size_metrics
from .summarization import build_paper_metric_report, compare_memory_qa_results, summarize_memory_qa


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
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--case-graph-baseline",
        dest="case_graph_baseline",
        action="store_true",
        default=False,
        help=(
            "Also answer every target from the original target-safe CaseGraph entity/edge "
            "memory and report a paired comparison. Requires --graphs."
        ),
    )
    baseline_group.add_argument(
        "--no-case-graph-baseline",
        dest="case_graph_baseline",
        action="store_false",
    )
    parser.add_argument("--coverage-dir", help="Defaults to the sibling coverage_states directory.")
    parser.add_argument(
        "--require-certified-memory",
        action="store_true",
        help="Do not answer cases whose coverage state is absent or not marked done.",
    )
    parser.add_argument("--output", help="Defaults to <memory-dir>/../memory_qa_results.json.")
    parser.add_argument(
        "--details-output",
        help=(
            "Full per-case diagnostics. Defaults to <output-stem>.details.json; --output "
            "contains only the compact paper-metric report."
        ),
    )
    parser.add_argument("--case-id", action="append", default=[], help="Evaluate only this case id; repeatable.")
    parser.add_argument("--start-index", type=int, default=1, help="One-based index into selected cases.")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse case results already present in --output.")
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--training-config", default="configs/online_grpo.yaml")
    parser.add_argument(
        "--answer-top-k",
        type=int,
        help="Number of compressed memory values given to the answer model (paper Value=Key default: 20).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Deprecated alias for --answer-top-k.",
    )
    parser.add_argument(
        "--retrieval-k-values",
        default="5,10,20,30",
        help="Comma-separated retrieval cutoffs used by the released UnifiedMem evaluator.",
    )
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

    parser.add_argument(
        "--judge-mode",
        choices=("deepseek", "llm", "string"),
        default=os.environ.get("JUDGE_MODE", "deepseek"),
    )
    parser.add_argument("--judge-api-base", default=os.environ.get("JUDGE_API_BASE"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--judge-max-output-tokens", type=int, default=10)
    parser.add_argument(
        "--judge-thinking",
        choices=("disabled", "enabled", "server_default"),
        default=(
            os.environ.get("JUDGE_THINKING")
            or os.environ.get("DEEPSEEK_THINKING")
            or "disabled"
        ),
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
    if args.case_graph_baseline and not args.graphs:
        raise SystemExit("--case-graph-baseline requires --graphs.")

    memory_dir = Path(args.memory_dir)
    output_path = (
        Path(args.output)
        if args.output
        else memory_dir.parent / "memory_qa_results.json"
    )
    details_output_path = (
        Path(args.details_output)
        if args.details_output
        else output_path.with_name(f"{output_path.stem}.details{output_path.suffix or '.json'}")
    )
    coverage_dir = (
        Path(args.coverage_dir)
        if args.coverage_dir
        else memory_dir.parent / "coverage_states"
    )

    dataset_questions = load_longmemeval_questions(args.dataset)
    case_graphs = load_case_graphs(args.graphs) if args.graphs else {}
    graph_questions = {
        case_id: MemoryQuestion.from_graph(graph)
        for case_id, graph in case_graphs.items()
    }
    questions = merge_question_metadata(dataset_questions, graph_questions)
    memory_case_ids = discover_memory_case_ids(memory_dir)
    memory_case_id_set = set(memory_case_ids)
    graph_case_id_set = set(graph_questions)
    graph_cases_without_memory: List[str] = []
    memory_cases_without_graph: List[str] = []
    if args.case_graph_baseline:
        memory_cases_without_graph = sorted(memory_case_id_set - graph_case_id_set)
        if memory_cases_without_graph:
            raise SystemExit(
                "CaseGraph baseline requires a matching graph for every memory case. "
                "Missing graph case ids: "
                + ", ".join(memory_cases_without_graph[:20])
            )
        graph_cases_without_memory = sorted(graph_case_id_set - memory_case_id_set)
        selected_ids = sorted(memory_case_id_set)
        if graph_cases_without_memory:
            logger.info(
                "CaseGraph baseline will ignore %d graph cases without constructed memory; "
                "both methods will evaluate the same %d memory-backed cases.",
                len(graph_cases_without_memory),
                len(selected_ids),
            )
    elif graph_questions:
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

    if (
        args.answer_top_k is not None
        and args.top_k is not None
        and args.answer_top_k != args.top_k
    ):
        raise SystemExit("--answer-top-k and --top-k disagree; set only one value.")
    retrieval_k_values = _parse_positive_ints(args.retrieval_k_values)
    if not retrieval_k_values:
        raise SystemExit("--retrieval-k-values must contain at least one positive integer.")

    training_config = _load_training_config(args.training_config)
    answer_top_k = int(
        args.answer_top_k
        if args.answer_top_k is not None
        else args.top_k if args.top_k is not None else 20
    )
    if answer_top_k < 1:
        raise SystemExit("--answer-top-k must be >= 1.")
    top_k_points = int(_setting(args.top_k_points, training_config, "top_k_points", 32))
    min_score = float(args.min_score if args.min_score is not None else -2.0)
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
    if args.judge_mode in {"deepseek", "llm"}:
        judge_client = _build_judge_client(args, answer_client)
        if not args.skip_endpoint_preflight and judge_client is not answer_client:
            _preflight_endpoint(judge_client, "judge")
    answer_agent = LongMemEvalMemoryAnswerAgent(
        client=answer_client,
        max_output_tokens=args.answer_max_output_tokens,
    )
    if args.judge_mode in {"deepseek", "llm"}:
        judge = LongMemEvalAnswerJudge(
            client=judge_client,
            max_output_tokens=args.judge_max_output_tokens,
            provider=args.judge_mode,
        )
    else:
        judge = AnswerEquivalenceJudge(use_llm=False)

    base_payload = {
        "schema_version": 3,
        "dataset": str(args.dataset),
        "graphs": str(args.graphs or ""),
        "memory_dir": str(memory_dir),
        "coverage_dir": str(coverage_dir),
        "unmatched_memory_case_ids": unmatched_memory_ids,
        "case_alignment": {
            "selection_rule": (
                "constructed_memory_cases_with_required_matching_graph"
                if args.case_graph_baseline
                else "graph_denominator" if graph_questions else "memory_files"
            ),
            "memory_case_count": len(memory_case_id_set),
            "graph_case_count": len(graph_case_id_set),
            "graph_cases_without_memory": graph_cases_without_memory,
            "memory_cases_without_graph": memory_cases_without_graph,
        },
        "config": {
            "answer_api_base": answer_client.base_url,
            "answer_model": answer_client.model,
            "answer_thinking": args.answer_thinking,
            "answer_max_output_tokens": args.answer_max_output_tokens,
            "judge_mode": args.judge_mode,
            "judge_api_base": judge_client.base_url if judge_client is not None else "",
            "judge_model": judge_client.model if judge_client is not None else "",
            "judge_thinking": (
                args.answer_thinking
                if judge_client is answer_client
                else args.judge_thinking
            ),
            "judge_max_output_tokens": args.judge_max_output_tokens,
            "retriever": retriever_config,
            "answer_top_k": answer_top_k,
            "retrieval_k_values": retrieval_k_values,
            "top_k_points": top_k_points,
            "min_score": min_score,
            "require_certified_memory": args.require_certified_memory,
            "case_graph_baseline": args.case_graph_baseline,
            "case_graph_baseline_definition": (
                "target-safe original graph entities and relationships; no raw sessions"
            ),
            "paper_protocol": {
                "benchmark": "LongMemEval",
                "memory_value_type": "Key",
                "primary_retrieval_metrics": ["R@5", "R@10", "N@5", "N@10"],
                "primary_overall_metric": "answer_accuracy",
            },
        },
    }
    resume_path = details_output_path if details_output_path.exists() else output_path
    resume_allowed = args.resume and _resume_compatible(resume_path, base_payload)
    prior_results = _load_prior_results(resume_path) if resume_allowed else []
    stale_prior_count = sum(
        "longmemeval_retrieval_metrics" not in item for item in prior_results
    )
    if stale_prior_count:
        logger.warning(
            "Ignoring %d resumed results written before the paper-metric schema.",
            stale_prior_count,
        )
    selected_id_set = set(selected_ids)
    results_by_id = {
        str(item.get("case_id")): item
        for item in prior_results
        if str(item.get("case_id")) in selected_id_set
        and "longmemeval_retrieval_metrics" in item
    }
    prior_baseline_results = (
        _load_prior_baseline_results(resume_path)
        if resume_allowed and args.case_graph_baseline
        else []
    )
    baseline_results_by_id = {
        str(item.get("case_id")): item
        for item in prior_baseline_results
        if str(item.get("case_id")) in selected_id_set
        and "longmemeval_retrieval_metrics" in item
    }
    logger.info(
        "Evaluating %d cases: answer_model=%s retriever=%s model=%s device=%s answer_top_k=%d retrieval_k=%s",
        len(selected_ids),
        answer_client.model,
        retriever_config.get("type"),
        retriever_config.get("model_name"),
        retriever_config.get("device") or "auto",
        answer_top_k,
        retrieval_k_values,
    )
    if args.case_graph_baseline:
        logger.info(
            "CaseGraph baseline enabled: using the same retriever, top-k, answer model, "
            "and judge over target-safe entity/relationship values."
        )

    completed_now = 0
    for index, case_id in enumerate(selected_ids, start=1):
        method_pending = case_id not in results_by_id
        baseline_pending = args.case_graph_baseline and case_id not in baseline_results_by_id
        if not method_pending and not baseline_pending:
            logger.info("[%d/%d] %s resumed", index, len(selected_ids), case_id)
            continue
        question = questions[case_id]
        if method_pending:
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
                    retrieval_k_values=retrieval_k_values,
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
                        retrieval_k_values=retrieval_k_values,
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
                            retrieval_k_values=retrieval_k_values,
                        )
                    else:
                        try:
                            result = evaluate_memory_question(
                                question,
                                memory_store,
                                answer_agent,
                                judge,
                                retriever_config=retriever_config,
                                top_k=answer_top_k,
                                retrieval_k_values=retrieval_k_values,
                                min_score=min_score,
                                memory_path=str(memory_path),
                                coverage_state=coverage_state,
                            )
                        except Exception as exc:
                            logger.exception("Case %s refactored-memory evaluation failed", case_id)
                            result = failed_result(
                                question,
                                status="evaluation_error",
                                reason=str(exc),
                                memory_path=str(memory_path),
                                memory_store=memory_store,
                                coverage_state=coverage_state,
                                retrieval_k_values=retrieval_k_values,
                            )
            result["memory_source"] = "refactored_memory"
            results_by_id[case_id] = result
        else:
            result = results_by_id[case_id]

        baseline_result: Optional[Dict[str, Any]] = baseline_results_by_id.get(case_id)
        if baseline_pending:
            graph = case_graphs.get(case_id)
            graph_label = _case_graph_label(args.graphs, case_id)
            if graph is None:
                baseline_result = failed_result(
                    question,
                    status="missing_case_graph",
                    reason=f"No CaseGraph found for {case_id}.",
                    memory_path=graph_label,
                    retrieval_k_values=retrieval_k_values,
                )
            else:
                graph_memory = case_graph_to_memory_store(graph)
                try:
                    baseline_result = evaluate_memory_question(
                        question,
                        graph_memory,
                        answer_agent,
                        judge,
                        retriever_config=retriever_config,
                        top_k=answer_top_k,
                        retrieval_k_values=retrieval_k_values,
                        min_score=min_score,
                        memory_path=graph_label,
                    )
                except Exception as exc:
                    logger.exception("Case %s CaseGraph baseline evaluation failed", case_id)
                    baseline_result = failed_result(
                        question,
                        status="baseline_evaluation_error",
                        reason=str(exc),
                        memory_path=graph_label,
                        memory_store=graph_memory,
                        retrieval_k_values=retrieval_k_values,
                    )
            baseline_result["memory_source"] = "case_graph"
            baseline_retrieval = baseline_result.get("longmemeval_retrieval_metrics")
            if isinstance(baseline_retrieval, dict):
                baseline_retrieval["adaptation"] = (
                    "case_graph_value_with_answer_session_provenance"
                )
            baseline_results_by_id[case_id] = baseline_result

        completed_now += 1
        ordered_results = [results_by_id[item] for item in selected_ids if item in results_by_id]
        running = summarize_memory_qa(ordered_results)
        logger.info(
            "[%d/%d] %s status=%s correct=%s running_score=%.4f R@5=%s N@5=%s",
            index,
            len(selected_ids),
            case_id,
            result.get("status"),
            result.get("correct"),
            running["overall_score"],
            _format_metric(
                (result.get("longmemeval_retrieval_metrics") or {}).get("recall_all@5")
            ),
            _format_metric(
                (result.get("longmemeval_retrieval_metrics") or {}).get("ndcg_any@5")
            ),
        )
        if args.case_graph_baseline and baseline_result is not None:
            ordered_baselines = [
                baseline_results_by_id[item]
                for item in selected_ids
                if item in baseline_results_by_id
            ]
            baseline_running = summarize_memory_qa(ordered_baselines)
            logger.info(
                "[%d/%d] %s case_graph_status=%s correct=%s "
                "running_case_graph_score=%.4f paired_delta=%.4f",
                index,
                len(selected_ids),
                case_id,
                baseline_result.get("status"),
                baseline_result.get("correct"),
                baseline_running["overall_score"],
                running["overall_score"] - baseline_running["overall_score"],
            )
        if completed_now % args.save_every == 0:
            _write_output(
                output_path,
                details_output_path,
                base_payload,
                selected_ids,
                results_by_id,
                baseline_results_by_id if args.case_graph_baseline else None,
            )

    payload = _write_output(
        output_path,
        details_output_path,
        base_payload,
        selected_ids,
        results_by_id,
        baseline_results_by_id if args.case_graph_baseline else None,
    )
    print(json.dumps(payload["paper_metrics"], ensure_ascii=False, indent=2))
    print(f"Paper metrics: {output_path}")
    print(f"Detailed per-case results: {details_output_path}")


def _build_judge_client(
    args: argparse.Namespace,
    answer_client: OpenAIChatClient,
) -> OpenAIChatClient:
    if args.judge_mode == "deepseek":
        api_key = args.judge_api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "JUDGE_MODE=deepseek requires DEEPSEEK_API_KEY or JUDGE_API_KEY."
            )
        thinking = (
            None
            if args.judge_thinking == "server_default"
            else {"type": args.judge_thinking}
        )
        return OpenAIChatClient(
            model=(
                args.judge_model
                or os.environ.get("DEEPSEEK_MODEL")
                or "deepseek-v4-flash"
            ),
            api_key=api_key,
            base_url=(
                args.judge_api_base
                or os.environ.get("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com"
            ).rstrip("/"),
            timeout=args.judge_timeout,
            thinking=thinking,
        )
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


def _parse_positive_ints(value: str) -> List[int]:
    try:
        return sorted(
            {
                int(item.strip())
                for item in str(value or "").split(",")
                if item.strip() and int(item.strip()) > 0
            }
        )
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --retrieval-k-values {value!r}; expected comma-separated integers."
        ) from exc


def _format_metric(value: Any) -> str:
    if value is None:
        return "excluded"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "invalid"


def _case_graph_label(graphs: Optional[str], case_id: str) -> str:
    if not graphs:
        return f"case_graph:{case_id}"
    path = Path(graphs)
    if path.is_dir():
        return str(path / f"{case_id}.case_graph.json")
    return str(path)


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


def _load_prior_baseline_results(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json_if_object(path)
    baseline = payload.get("baseline", {})
    results = baseline.get("results", []) if isinstance(baseline, dict) else []
    if not isinstance(results, list):
        return []
    logger.info("Loaded %d prior CaseGraph baseline results from %s", len(results), path)
    return [item for item in results if isinstance(item, dict)]


def _resume_compatible(path: Path, current: Dict[str, Any]) -> bool:
    if not path.exists():
        return True
    prior = _load_json_if_object(path)
    if prior.get("schema_version") != current.get("schema_version"):
        logger.warning(
            "Ignoring resumed results because schema_version changed from %r to %r.",
            prior.get("schema_version"),
            current.get("schema_version"),
        )
        return False

    prior_config = prior.get("config") if isinstance(prior.get("config"), dict) else {}
    current_config = (
        current.get("config") if isinstance(current.get("config"), dict) else {}
    )
    compared_keys = (
        "answer_api_base",
        "answer_model",
        "answer_thinking",
        "answer_max_output_tokens",
        "judge_mode",
        "judge_api_base",
        "judge_model",
        "judge_thinking",
        "judge_max_output_tokens",
        "retriever",
        "answer_top_k",
        "retrieval_k_values",
        "min_score",
    )
    changed = [
        key for key in compared_keys if prior_config.get(key) != current_config.get(key)
    ]
    if changed:
        logger.warning(
            "Ignoring resumed results because evaluation settings changed: %s.",
            ", ".join(changed),
        )
        return False
    return True


def _write_output(
    output_path: Path,
    base_payload: Dict[str, Any],
    selected_ids: List[str],
    results_by_id: Dict[str, Dict[str, Any]],
    baseline_results_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    results = [results_by_id[case_id] for case_id in selected_ids if case_id in results_by_id]
    payload = {
        **base_payload,
        "selected_case_ids": selected_ids,
        "summary": summarize_memory_qa(results),
        "results": results,
    }
    if baseline_results_by_id is not None:
        baseline_results = [
            baseline_results_by_id[case_id]
            for case_id in selected_ids
            if case_id in baseline_results_by_id
        ]
        baseline_summary = summarize_memory_qa(baseline_results)
        baseline_summary["paper_metrics"]["retrieval_unit"] = (
            "case_graph_entity_or_relationship"
        )
        payload["baseline"] = {
            "name": "case_graph",
            "definition": (
                "Original CaseGraph entity and relationship values after removing target "
                "metadata and evaluator-injected answer safeguards. Raw sessions are not used."
            ),
            "summary": baseline_summary,
            "results": baseline_results,
        }
        payload["comparison"] = compare_memory_qa_results(results, baseline_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    main()
