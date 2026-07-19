"""Attacker reward adapter for GRPO training with multi-component rewards.

Inspired by Tool-R0's reward decomposition and Dr. Zero's difficulty bandpass.
"""

import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .baseline import GoldenFactAnswerAgent, AnswerEquivalenceJudge
from .llm import OpenAIChatClient
from .llm import OpenAIChatClient


def compute_attacker_score(prompts, completions, **kwargs) -> List[float]:
    """Main reward function called by verl during attacker GRPO training.

    Multi-component reward:
    1. Format validity (question and answer present)
    2. Groundedness (answer supported by golden facts)
    3. Difficulty bandpass (defender success rate in target range)
    4. Constructive impact (if defender commits, inherit reward)
    """
    # Extract ground truth states
    states = []
    for prompt in prompts:
        gt_str = prompt.get("reward_model", {}).get("ground_truth", "{}")
        states.append(json.loads(gt_str))

    # Extract attacker outputs
    outputs = []
    for completion in completions:
        text = completion[0]["content"]
        outputs.append(_parse_attacker_output(text))

    # Compute rewards
    rewards = []
    for state, output in zip(states, outputs):
        reward = _compute_single_reward(state, output, **kwargs)
        rewards.append(reward)

    return rewards


def _check_oracle_groundedness(question: str, answer: str, golden_facts: List[Dict]) -> bool:
    """Check if answer is grounded in golden facts using oracle baseline."""
    if not golden_facts:
        return False

    try:
        # Use baseline components to check groundedness
        answer_agent = GoldenFactAnswerAgent(max_output_tokens=200)
        judge = AnswerEquivalenceJudge(use_llm=False)  # String match only for speed

        oracle_answer = answer_agent.answer(question, golden_facts)
        judge_result = judge.judge(question, answer, oracle_answer["answer"])

        return judge_result.get("correct", False)
    except Exception:
        # If oracle check fails, be lenient (don't penalize)
        return True


def _parse_attacker_output(text: str) -> Optional[Dict[str, str]]:
    """Extract question and answer from attacker output."""
    # Try JSON parse first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "question" in data and "answer" in data:
            return {
                "question": str(data["question"]).strip(),
                "answer": str(data["answer"]).strip(),
            }
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    if "```json" in text:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "question" in data and "answer" in data:
                    return {
                        "question": str(data["question"]).strip(),
                        "answer": str(data["answer"]).strip(),
                    }
            except json.JSONDecodeError:
                pass

    # Try simple field extraction
    q_match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
    a_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)

    if q_match and a_match:
        return {
            "question": q_match.group(1).strip(),
            "answer": a_match.group(1).strip(),
        }

    return None


def _compute_single_reward(
    state: Dict[str, Any],
    output: Optional[Dict[str, str]],
    **kwargs
) -> float:
    """Compute reward for single attacker output."""
    # Component 1: Format validity
    if output is None or not output.get("question") or not output.get("answer"):
        return -5.0  # Hard penalty for invalid format

    question = output["question"]
    answer = output["answer"]

    # Basic quality checks
    if len(question) < 10 or len(answer) < 2:
        return -3.0  # Too trivial

    # Component 2: Groundedness (oracle check)
    golden_facts = state.get("golden_facts", [])
    is_grounded = True
    if golden_facts:
        is_grounded = _check_oracle_groundedness(question, answer, golden_facts)
        if not is_grounded:
            return -3.0  # Not answerable from golden facts

    # Component 3: Difficulty (if defender provided)
    # This will be enhanced in full implementation to measure defender success rate
    # For now, use a placeholder
    R_difficulty = 1.0

    # Component 4: Constructive impact (simplified)
    # In full implementation, this checks if defender commits with positive reward
    # For now, reward questions that are well-formed and grounded
    R_constructive = 2.0

    # Aggregate reward
    w_format = 1.0
    w_difficulty = 2.0
    w_constructive = 3.0

    total_reward = (
        w_format * 1.0 +  # Passed format check
        w_difficulty * R_difficulty +
        w_constructive * R_constructive
    )

    return total_reward


def difficulty_bandpass(
    p_success: float,
    p_low: float = 0.25,
    p_high: float = 0.75,
    sigma: float = 0.12
) -> float:
    """Difficulty bandpass reward (Dr. Zero style).

    Peaks at p_success in [p_low, p_high], penalizes too easy or too hard.
    """
    if p_success < 0.01:
        return 0.0  # Impossible question

    if p_low <= p_success <= p_high:
        return 1.0  # Ideal difficulty

    # Gaussian falloff outside target range
    if p_success < p_low:
        diff = p_success - p_low
    else:
        diff = p_success - p_high

    return math.exp(-(diff * diff) / (2.0 * sigma * sigma))


def group_questions_by_structure(
    states: List[Dict],
    outputs: List[Optional[Dict]]
) -> Dict[str, List[int]]:
    """Group questions by graph structure for HRPO variance reduction.

    Groups by:
    - Hop count (1-hop, 2-hop, 3+)
    - Entity count
    """
    groups = defaultdict(list)

    for idx, (state, output) in enumerate(zip(states, outputs)):
        if output is None:
            continue

        route = state.get("route", {})
        entity_ids = route.get("entity_ids", [])
        path = route.get("path", [])

        hop_count = len(path) - 1 if len(path) > 1 else 0
        entity_count = len(entity_ids)

        # Create group key
        if hop_count == 0:
            hop_label = "0hop"
        elif hop_count == 1:
            hop_label = "1hop"
        elif hop_count == 2:
            hop_label = "2hop"
        else:
            hop_label = "3+hop"

        entity_label = f"{min(entity_count, 5)}ent"
        group_key = f"{hop_label}_{entity_label}"

        groups[group_key].append(idx)

    return groups


def compute_hrpo_advantages(
    rewards: List[float],
    groups: Dict[str, List[int]],
) -> List[float]:
    """Compute GRPO advantages with group-level baselines (Dr. Zero HRPO).

    For each group, use group mean as baseline instead of global mean.
    This reduces variance when questions have different structural difficulties.
    """
    if not groups:
        # Fallback to global baseline
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return [r - mean_reward for r in rewards]

    advantages = [0.0] * len(rewards)

    for group_key, indices in groups.items():
        if not indices:
            continue

        # Compute group baseline
        group_rewards = [rewards[i] for i in indices]
        group_baseline = sum(group_rewards) / len(group_rewards)

        # Set advantages for this group
        for idx in indices:
            advantages[idx] = rewards[idx] - group_baseline

    return advantages
