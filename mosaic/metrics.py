"""Metrics computed directly from structured ExampleResult records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schemas import ExampleResult


@dataclass
class IterationStats:
    iteration: int
    correct: int
    valid: int
    total: int

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    @property
    def validity(self) -> float:
        return 100.0 * self.valid / self.total if self.total else 0.0


@dataclass
class Metrics:
    total: int
    final_correct: int
    final_valid: int
    per_iteration: list[IterationStats]
    top_verifier_failures: list[tuple[str, int]]
    top_validator_failures: list[tuple[str, int]]

    @property
    def final_accuracy(self) -> float:
        return 100.0 * self.final_correct / self.total if self.total else 0.0

    @property
    def final_validity(self) -> float:
        return 100.0 * self.final_valid / self.total if self.total else 0.0


def _per_iteration_stats(results: list[ExampleResult]) -> list[IterationStats]:
    if not results:
        return []
    max_iters = max(len(r.iterations) for r in results) if any(r.iterations for r in results) else 0
    stats: list[IterationStats] = []
    for i in range(1, max_iters + 1):
        correct = 0
        valid = 0
        for r in results:
            seen_correct = any(
                it.verification.is_same_chemical for it in r.iterations[:i]
            )
            seen_valid = any(
                it.validation.is_valid_reactant for it in r.iterations[:i]
            )
            if seen_correct:
                correct += 1
            if seen_valid:
                valid += 1
        stats.append(IterationStats(iteration=i, correct=correct, valid=valid, total=len(results)))
    return stats


def _top_failures(results: list[ExampleResult], kind: str, k: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for r in results:
        final = r.final
        if final is None:
            continue
        if kind == "verifier" and not final.verification.is_same_chemical:
            counter[final.verification.explanation] += 1
        elif kind == "validator" and not final.validation.is_valid_reactant:
            counter[final.validation.explanation] += 1
    return counter.most_common(k)


def compute(results: list[ExampleResult]) -> Metrics:
    final_correct = sum(1 for r in results if r.is_correct)
    final_valid = sum(1 for r in results if r.is_valid)
    return Metrics(
        total=len(results),
        final_correct=final_correct,
        final_valid=final_valid,
        per_iteration=_per_iteration_stats(results),
        top_verifier_failures=_top_failures(results, "verifier"),
        top_validator_failures=_top_failures(results, "validator"),
    )


def format_summary(metrics: Metrics) -> str:
    lines = [
        "=== Verifier Metrics ===",
        f"Final accuracy: {metrics.final_accuracy:.2f}% "
        f"({metrics.final_correct}/{metrics.total})",
        "Per-iteration cumulative accuracy:",
    ]
    for s in metrics.per_iteration:
        lines.append(f"  iter {s.iteration}: {s.accuracy:.2f}% ({s.correct}/{s.total})")

    lines.append("")
    lines.append("=== Validator Metrics ===")
    lines.append(
        f"Final validity: {metrics.final_validity:.2f}% "
        f"({metrics.final_valid}/{metrics.total})"
    )
    lines.append("Per-iteration cumulative validity:")
    for s in metrics.per_iteration:
        lines.append(f"  iter {s.iteration}: {s.validity:.2f}% ({s.valid}/{s.total})")

    if metrics.top_validator_failures:
        lines.append("")
        lines.append("Top validator failure reasons:")
        for reason, count in metrics.top_validator_failures:
            lines.append(f"  {count}x — {reason}")

    if metrics.top_verifier_failures:
        lines.append("")
        lines.append("Top verifier failure reasons:")
        for reason, count in metrics.top_verifier_failures:
            lines.append(f"  {count}x — {reason}")

    return "\n".join(lines)
