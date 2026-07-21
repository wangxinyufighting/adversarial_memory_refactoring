#!/usr/bin/env python3
"""Compare evaluation results between our method and baseline.

Usage:
    python scripts/compare_methods.py outputs/memory_qa_comparisonv3.json
    python scripts/compare_methods.py outputs/memory_qa_comparisonv3.json --output outputs/comparison.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_comparison_data(file_path: str) -> Dict[str, Any]:
    """Load comparison JSON file."""
    with open(file_path) as f:
        return json.load(f)


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print()
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def compare_overall_performance(our_summary: Dict, baseline_summary: Dict):
    """Compare overall performance metrics."""
    print_section("📊 整体性能对比")

    print(f"{'指标':<30} {'我们的方法':<15} {'Baseline':<15} {'提升':<15}")
    print("-" * 75)

    our_acc = our_summary.get('accuracy', 0)
    baseline_acc = baseline_summary.get('accuracy', 0)
    improvement = our_acc - baseline_acc
    print(f"{'答案准确率':<30} {our_acc:<14.2%} {baseline_acc:<14.2%} {improvement:>+13.2%}")

    our_eval_acc = our_summary.get('evaluated_accuracy', 0)
    baseline_eval_acc = baseline_summary.get('evaluated_accuracy', 0)
    eval_improvement = our_eval_acc - baseline_eval_acc
    print(f"{'评估准确率':<30} {our_eval_acc:<14.2%} {baseline_eval_acc:<14.2%} {eval_improvement:>+13.2%}")


def compare_retrieval_metrics(our_summary: Dict, baseline_summary: Dict):
    """Compare retrieval performance."""
    print_section("🎯 检索性能对比")

    print(f"{'指标':<30} {'我们的方法':<15} {'Baseline':<15} {'提升':<15}")
    print("-" * 75)

    our_pm = our_summary.get('paper_metrics', {})
    baseline_pm = baseline_summary.get('paper_metrics', {})

    metrics = [
        ('R@5', 'R@5'),
        ('R@10', 'R@10'),
        ('N@5', 'N@5'),
        ('N@10', 'N@10'),
    ]

    for label, key in metrics:
        our_val = our_pm.get(key, 0)
        baseline_val = baseline_pm.get(key, 0)
        diff = our_val - baseline_val
        print(f"{label:<30} {our_val:<14.3f} {baseline_val:<14.3f} {diff:>+13.3f}")


def compare_memory_compression(our_summary: Dict, baseline_summary: Dict):
    """Compare memory compression efficiency."""
    print_section("💾 记忆压缩对比")

    print(f"{'指标':<30} {'我们的方法':<15} {'Baseline':<15} {'差异':<15}")
    print("-" * 75)

    our_mem = our_summary.get('memory', {})
    baseline_mem = baseline_summary.get('memory', {})

    our_chunks = our_mem.get('mean_chunks_per_case', 0)
    baseline_chunks = baseline_mem.get('mean_chunks_per_case', 0)
    print(f"{'平均块数/案例':<30} {our_chunks:<14.1f} {baseline_chunks:<14.1f} {our_chunks - baseline_chunks:>+13.1f}")

    our_tokens = our_mem.get('mean_tokens_per_case_approx', 0)
    baseline_tokens = baseline_mem.get('mean_tokens_per_case_approx', 0)
    print(f"{'平均tokens/案例':<30} {our_tokens:<14.0f} {baseline_tokens:<14.0f} {our_tokens - baseline_tokens:>+13.0f}")

    our_compression = our_mem.get('mean_char_reduction', 0)
    baseline_compression = baseline_mem.get('mean_char_reduction', 0)
    print(f"{'压缩率':<30} {our_compression:<14.2%} {baseline_compression:<14.2%} {our_compression - baseline_compression:>+13.2%}")

    # Compute compression ratio
    if our_tokens > 0:
        compression_ratio = baseline_tokens / our_tokens
        print()
        print(f"  ✅ 压缩比: {compression_ratio:.1f}x (节省 {(1-1/compression_ratio)*100:.1f}%的存储)")


def compare_by_question_type(our_summary: Dict, baseline_summary: Dict):
    """Compare performance by question type."""
    print_section("📋 按问题类型分析")

    our_by_type = our_summary.get('by_question_type', {})
    baseline_by_type = baseline_summary.get('by_question_type', {})

    all_types = sorted(set(list(our_by_type.keys()) + list(baseline_by_type.keys())))

    for qtype in all_types:
        our_stats = our_by_type.get(qtype, {})
        baseline_stats = baseline_by_type.get(qtype, {})

        our_acc = our_stats.get('accuracy', 0)
        baseline_acc = baseline_stats.get('accuracy', 0)
        diff = our_acc - baseline_acc

        our_total = our_stats.get('total', 0)
        baseline_total = baseline_stats.get('total', 0)

        print(f"  {qtype}:")
        print(f"    案例数: 我们 {our_total}, Baseline {baseline_total}")
        print(f"    准确率: 我们 {our_acc:.1%}, Baseline {baseline_acc:.1%}, 差异 {diff:+.1%}")


def print_key_findings(our_summary: Dict, baseline_summary: Dict):
    """Print key findings and conclusions."""
    print_section("🎯 核心发现")

    our_mem = our_summary.get('memory', {})
    baseline_mem = baseline_summary.get('memory', {})

    our_tokens = our_mem.get('mean_tokens_per_case_approx', 0)
    baseline_tokens = baseline_mem.get('mean_tokens_per_case_approx', 0)
    compression_ratio = baseline_tokens / our_tokens if our_tokens > 0 else 0

    our_pm = our_summary.get('paper_metrics', {})
    baseline_pm = baseline_summary.get('paper_metrics', {})

    r10_diff = our_pm.get('R@10', 0) - baseline_pm.get('R@10', 0)

    our_acc = our_summary.get('accuracy', 0)
    baseline_acc = baseline_summary.get('accuracy', 0)

    print("✅ 主要优势：")
    print(f"  • 记忆压缩：{compression_ratio:.1f}x (从 {baseline_tokens:.0f} 降至 {our_tokens:.0f} tokens/案例)")
    print(f"  • 存储节省：{(1-1/compression_ratio)*100:.1f}%")
    print()

    print("⚖️ 性能权衡：")
    print(f"  • 检索召回率下降：R@10 {r10_diff:+.1%}")
    print(f"  • 端到端准确率：{our_acc:.1%} vs {baseline_acc:.1%} (差异 {our_acc - baseline_acc:+.1%})")
    print()

    print("💡 结论：")
    if abs(our_acc - baseline_acc) < 0.02:  # Within 2%
        print(f"  在保持相同端到端准确率的同时，实现了{compression_ratio:.1f}x的记忆压缩。")
        print(f"  检索性能下降约{abs(r10_diff)*100:.0f}%是为大幅压缩付出的合理代价。")
    else:
        print(f"  实现了{compression_ratio:.1f}x的记忆压缩，但准确率有所变化。")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("comparison_file", help="Path to comparison JSON file")
    parser.add_argument("--output", help="Output markdown file (optional)")
    args = parser.parse_args()

    # Load data
    data = load_comparison_data(args.comparison_file)
    our_summary = data.get('summary', {})
    baseline_summary = data.get('baseline', {}).get('summary', {})

    if not baseline_summary:
        print("❌ Error: No baseline data found in comparison file")
        return 1

    # Print comparisons
    print_section(f"方法对比分析：{Path(args.comparison_file).name}")
    print(f"Memory: {data.get('memory_dir', 'N/A')}")
    print(f"Graphs: {data.get('graphs', 'N/A')}")
    print(f"Cases: {our_summary.get('total', 0)}")

    print_key_findings(our_summary, baseline_summary)
    compare_overall_performance(our_summary, baseline_summary)
    compare_retrieval_metrics(our_summary, baseline_summary)
    compare_memory_compression(our_summary, baseline_summary)
    compare_by_question_type(our_summary, baseline_summary)

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
