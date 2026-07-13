# Evaluation Memory Construction Optimization Report

## 1. Objective and invariants

The Evaluation builder must construct a fixed memory from raw sessions before the held-out target question is exposed to the construction policy. The optimized implementation enforces four invariants:

1. **No target leakage**: target question, answer, answer source IDs, and evaluator-injected graph facts are removed before routing and probe generation.
2. **Completeness before compactness**: an edit is not accepted merely because an answer token appears. The saved evidence must retain the relevant subject, relation, object, temporal qualifiers, and, for aggregate probes, every supporting event.
3. **Compact persisted state**: raw dialogue is not hidden in `metadata.facts`. Model-owned IDs and source provenance are not trusted.
4. **Fail closed**: target evaluation does not silently score a memory whose construction coverage was never certified.

## 2. Root causes found in the previous Evaluation run

The `00ca467f` run exposed several independent failures rather than one bad checkpoint:

- The graph contained 427 coverage units, but construction stopped after a fixed 250 questions. Only 55 units were covered, so the case could not be complete by design.
- Coverage mode mechanically cycled through candidates. A failed Dr. Thompson date probe was not retried immediately.
- Probes were one-edge questions. They could preserve March 3 and March 20 separately, but never tested whether memory could support the cross-session count `2`.
- Date matching treated `2023/03/20` and `March 20th, 2023` as different answers. Professional aliases also treated `ORTHOPEDIC SURGEON DR. THOMPSON` and `Dr. Thompson` as only a partial match.
- A defender response could place the correct fact in `summary` while leaving unrelated text in `content`; the proposal was then rolled back even though the model had produced useful evidence.
- The model could reuse memory IDs and emit incorrect source IDs. Dense retrieval grouped by ID, so duplicate IDs could silently collapse different chunks.
- Model metadata could retain an entire raw dialogue under `facts`. That made the reported `content` look compact while the actual retrieval state was not compressed.
- A malformed proposal produced two trace records (`proposal_failed` and `rolled_back`) for one question, inflating episode counts.
- Target evaluation excluded missing/incomplete cases from the accuracy denominator, which could make a failed construction run look better than it was.

These issues mean that the old 32-chunk output does not by itself prove the defender checkpoint was poorly trained. The Evaluation controller prevented the checkpoint from receiving enough retries, compositional tests, and normalized reward signals.

## 3. New construction algorithm

### 3.1 Target-free coverage model

Each public relationship is classified as either memory-worthy or background. User-centered, personal, episodic, temporal, and health/event facts are required. Generic background relations remain visible in `structural_coverage` but do not force irrelevant world knowledge into the final memory.

The stopping metric is `required_coverage`; `critical_coverage` must also reach its threshold. `structural_coverage` remains a diagnostic over every public graph edge.

### 3.2 Dynamic budget and adaptive scheduling

For coverage mode, the resolved question budget is:

```text
max(EPISODES_PER_CASE,
    required_units * QUESTIONS_PER_UNIT + CERTIFICATION_QUESTIONS)
```

It is capped by `HARD_MAX_QUESTIONS_PER_CASE`. The defaults are 250, 3, 60, and 2000 respectively. The run stops early after certification, so the resolved budget is a safety bound rather than a required number of calls.

Uncovered critical units are selected before background units. A failed unit is retried immediately, up to `MAX_RETRIES_PER_UNIT=3`. If an unresolved required unit exhausts its retries, the case ends as incomplete instead of spending calls on unrelated facts.

### 3.3 Session-aware compaction

The Evaluation-only defender prompt asks the policy to preserve all distinct personal, episodic, and temporal facts supported by the current raw session, not only the shortest answer phrase. After a commit, sibling graph units sharing the same source session are marked covered only when the persisted retrieval fields explicitly contain their subject, relation, and object. This can reduce repeated calls without assuming that a narrow summary preserved facts it did not state.

### 3.4 Compositional probes

Coverage mode now derives target-free temporal count probes from completed graph events. For example, two independently completed medical visits in March produce a probe asking for the number of completed visits in that month. Planned or upcoming events are excluded unless the graph evidence also establishes completion. It also groups repeated user relations across source sessions, producing probes such as the number of distinct instruments linked by `owns`; this catches cross-session retrieval failures beyond the medical example.

Aggregate completeness requires:

- a count that is either explicit or derivable from the complete atomic events;
- the requested time bucket;
- every supporting event identity/relation;
- every supporting date.

The aggregate itself is a separate critical coverage unit. Passing its atomic event probes cannot hide a failed aggregation probe.

### 3.5 Policy output normalization

The server call now uses 4096 output tokens, aligned with the training response scale, and retries malformed JSON twice. Before sandbox evaluation, the controller:

- promotes the most route-complete field among `content`, `summary`, and compact facts;
- assigns a deterministic, case/step-owned unique memory ID;
- overwrites source provenance from trusted golden session IDs;
- carries selected old-source provenance through Merge;
- removes raw `facts` payloads and caps compact retrieval metadata;
- preserves linked-question lineage through the existing settlement path.

This specifically repairs the old pattern where the useful Dr. Thompson date appeared only in `summary` while unrelated cough text occupied `content`.

### 3.6 Reward and normalization

Structured reward now normalizes ISO/slash dates and English month dates. It also handles professional title aliases and exact numeric tokens, so answer `2` no longer matches the `2` inside `2023`.

Aggregate probes always use the retrieved-memory answer agent and judge, even when the ordinary reward mode is `semantic_complete`. Atomic probes continue to require both answer correctness and structured completeness before they can advance coverage.

An initial-defense answer is no longer accepted on answer correctness alone. Its retrieved chunks pass the same completeness audit before the question enters the success pool. Thus a chunk containing only "two appointments" cannot certify a case without the two dated events.

### 3.7 Certification and target evaluation gate

After required coverage is reached, the builder runs consecutive certification probes. A case is usable only when `completion_status` is `done` and `stop_reason` is `certified`.

Any memory commit during certification invalidates the earlier certification streak and per-unit certification counts. The repaired memory must establish a fresh consecutive streak; otherwise certification would mix evidence from different versions of memory.

Target evaluation automatically reads the sibling `coverage_states` directory. Missing or incomplete construction states become end-to-end failures. The output reports:

- `accuracy`: correct answers divided by every requested case;
- `evaluated_accuracy`: correct answers divided by cases that reached target answering;
- `status_counts`: evaluated, missing memory, missing coverage state, and incomplete memory counts.

`--allow-incomplete-memory` exists only for diagnosis.

## 4. Performance changes

- Dense point embeddings are cached across repeated sandbox retriever rebuilds.
- Duplicate memory IDs no longer collapse physical chunks during dense retrieval.
- Coverage mode selects its adaptive candidate directly instead of performing and then discarding an unrelated random walk.
- Aggregate probes use the answer LLM only for the current compositional question; semantic-mode regression checks remain deterministic instead of multiplying API calls by the regression sample size.
- Compact traces omit repeated temporary memory stores while retaining the selected proposal, current-test diagnostics, reward, and regression summary.
- The fixed inference artifact drops construction-only linked questions, step numbers, and prompt text after certification; the separate success pool and trace retain debugging lineage.
- `CASE_WORKERS` enables case-level concurrency so vLLM can batch requests. Start with 2 workers on one serving GPU and increase only after checking GPU memory and latency.
- Proposal JSON retries avoid losing a coverage unit to transient truncation or formatting errors.

## 5. Output interpretation

For each case, inspect `coverage_states/<case_id>.json` first:

- `required_coverage`: stopping coverage over memory-worthy facts;
- `critical_coverage`: unweighted required-unit pass ratio;
- `structural_coverage`: all public graph relationships, including background;
- `consecutive_certification_passes`: current certification streak;
- `certification_resets`: memory changes that invalidated an in-progress streak;
- `question_budget`: resolved safety budget;
- `stop_reason`: `certified`, `question_budget_exhausted`, `coverage_candidates_exhausted`, `attack_failure_limit`, or `certification_budget_exhausted`.

The construction summary also reports `memory_payload_chars`, unique IDs, duplicate IDs, and top proposal/attack errors. A healthy formal run should have zero duplicate IDs and all benchmarked cases certified.

Construction and target evaluation both load `configs/online_grpo.yaml` by default. Unless a CLI/environment override is supplied, `top_k`, `top_k_points`, dense retriever model/device/mode/max length, thresholds, seed, and routing settings therefore match training. The resolved values are persisted in `construction_config.yaml`.

## 6. Recommended formal settings

```bash
ATTACKER_MODE=coverage \
EPISODES_PER_CASE=250 \
QUESTIONS_PER_UNIT=3 \
HARD_MAX_QUESTIONS_PER_CASE=2000 \
MAX_RETRIES_PER_UNIT=3 \
CERTIFICATION_QUESTIONS=60 \
DEFENDER_MAX_OUTPUT_TOKENS=4096 \
DEFENDER_PROPOSAL_RETRIES=2 \
TOP_K=8 \
TOP_K_POINTS=32 \
RETRIEVER_TYPE=dense_structured \
RETRIEVER_MODEL_NAME=/mnt/local2/wxy/models/contriever \
RETRIEVER_REQUIRE_MODEL=true \
TRACE_DETAIL=compact \
CASE_WORKERS=2 \
./scripts/construct_defender_memory_for_eval.sh
```

Do not set `SKIP_LLM_JUDGE=true` for the main result. Do not lower the hard cap merely to make the run finish; an incomplete coverage state must remain visible as a construction failure.

## 7. Remaining limits

Target-free construction cannot mathematically guarantee correctness for every possible natural-language question. `done` is an operational certificate over the public graph units and generated compositional probe families, not proof over all future questions. The current compositional generator covers repeated user-relation counts and completed temporal events; additional benchmark question families such as numeric sums, ordering, duration, comparison, and updates should be added as explicit probe types when observed.

Old memory files are not repaired in place. They must be rebuilt with the new constructor because their traces, IDs, provenance, raw metadata, and coverage state were produced under the previous logic.
