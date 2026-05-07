"""Compare validator models on a fixed solver output.

Per example: ONE solver call → one image X. Then ONE verifier call (with the
held-out reference D) gives ground truth, and N validator calls (no reference,
just A/B/C/X) give each candidate validator's judgment. Use the resulting
matrix to ask: which validator best predicts the verifier? How much do
validators (dis)agree among themselves?

This is intentionally NOT the full MOSAIC pipeline — there is no iteration,
no feedback loop. Single shot, multi-validator.

Outputs under runs/validators_<ts>/:
  - results.json    : per-example structured records (validator + verifier verdicts)
  - decisions.csv   : wide-format decision matrix (one row per example)
  - summary.txt     : per-validator agreement vs verifier (TP/FP/TN/FN, etc.)
  - report.html     : summary + per-example table with images
  - x_<id>.png      : the solver output image used for all evaluators on that example
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import html
import io
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import PIL.Image
from tqdm import tqdm

from mosaic import prompts
from mosaic.agents import (
    Backend,
    Solver,
    Validator,
    Verifier,
    is_gemini_direct,
    make_backend,
    pick_provider,
)
from mosaic.data import ReactionExample, load_examples
from mosaic.models import (
    DEFAULT_VERIFIER_OPENROUTER,
    OPENROUTER_BASE_URL,
    VALIDATOR_MODELS_OPENROUTER,
)
from mosaic.pipeline import _build_solver_contents
from mosaic.schemas import ValidationResult, VerificationResult


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "release_data", "eval")
_DEFAULT_OPENROUTER_KEY_FILE = os.path.join(_REPO_ROOT, ".secrets", "openrouter_api_key")
_DEFAULT_GOOGLE_KEY_FILE = os.path.join(_REPO_ROOT, ".secrets", "google_api_key")
_DEFAULT_SOLVER = "gemini-3-pro-image-preview"  # bare name → Gemini direct API
# Diverse, paid-tier subset spanning families; override with --validator-models.
_DEFAULT_VALIDATORS = [
    # Curated from a "Top 20 image-input" leaderboard. Bare names route to
    # GeminiBackend (Google AI Studio); slash-prefixed names route to
    # OpenRouter. See pick_provider() for the rule.

    # Google AI Studio (3 — Gemini + Gemma):
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",

    # OpenRouter (9):
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "moonshotai/kimi-k2.6",
    "x-ai/grok-4.3",
    "qwen/qwen3.5-397b-a17b",
]


def _safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


# ---------------------------------------------------------------------------
# Per-example record
# ---------------------------------------------------------------------------


@dataclass
class ValidatorVerdict:
    model: str
    is_valid: Optional[bool]
    explanation: str
    error: Optional[str] = None


@dataclass
class ExampleRecord:
    example_id: str
    difficulty: Optional[str]
    input_paths: list[str]
    solver_image_path: Optional[str]
    solver_error: Optional[str]
    verifier_correct: Optional[bool]
    verifier_explanation: str
    validator_verdicts: list[ValidatorVerdict]


def _difficulty_of(example: ReactionExample) -> Optional[str]:
    parent = os.path.dirname(os.path.dirname(example.image_a))
    return os.path.basename(parent) or None


# ---------------------------------------------------------------------------
# Per-example execution
# ---------------------------------------------------------------------------


def _run_one_example(
    *,
    example: ReactionExample,
    solver: Optional[Solver],
    verifier: Optional[Verifier],
    validators: list[Validator],
    out_dir: str,
    reuse_solver_from: Optional[str],
    reuse_iter: int,
    evaluators_per_example: int,
    log,
) -> ExampleRecord:
    """Solver once, then verifier + every validator on the same X (in parallel).

    If ``reuse_solver_from`` is set, skip the solver call and load the X image
    from disk: ``<reuse_solver_from>/iter_images/<example_id>/iter_<N>.png``.
    """
    rec = ExampleRecord(
        example_id=example.example_id,
        difficulty=_difficulty_of(example),
        input_paths=example.all_paths,
        solver_image_path=None,
        solver_error=None,
        verifier_correct=None,
        verifier_explanation="",
        validator_verdicts=[],
    )

    try:
        image_a = PIL.Image.open(example.image_a)
        image_b = PIL.Image.open(example.image_b)
        image_c = PIL.Image.open(example.image_c)
        reference_d = PIL.Image.open(example.image_d)
    except Exception as exc:
        rec.solver_error = f"image load failed: {exc}"
        log(f"[{example.example_id}] image load failed: {exc}")
        return rec

    if reuse_solver_from:
        src = os.path.join(
            reuse_solver_from, "iter_images", example.example_id,
            f"iter_{reuse_iter}.png",
        )
        if not os.path.exists(src):
            rec.solver_error = f"reuse: {src} not found"
            log(f"[{example.example_id}] {rec.solver_error}")
            return rec
        try:
            # Read the raw bytes from disk and decode through BytesIO so the
            # resulting PIL.Image has NO reference to the file handle. PIL's
            # default lazy loading shares state that breaks under concurrent
            # reads (manifests as "I/O operation on closed file", "broken PNG
            # file", etc. when multiple evaluator threads access the same
            # image). The .copy() after .load() additionally detaches the
            # decoded buffer from the BytesIO so threads can safely re-encode
            # in parallel.
            with open(src, "rb") as f:
                raw_bytes = f.read()
            opened = PIL.Image.open(io.BytesIO(raw_bytes))
            opened.load()
            x_image = opened.copy()
        except Exception as exc:
            rec.solver_error = f"reuse: failed to open {src}: {exc}"
            log(f"[{example.example_id}] {rec.solver_error}")
            return rec
        rec.solver_image_path = src  # absolute path; HTML report can b64-embed it
    else:
        assert solver is not None, "solver must be provided when not reusing"
        # Solver — single shot, no few-shots, no correction block.
        contents, _ = _build_solver_contents(
            prompts.SOLVER_PROMPT, image_a, image_b, image_c,
            few_shot_parts=[], correction_parts=[], correction_text="",
        )
        try:
            x_image = solver.run(contents)
        except Exception as exc:
            rec.solver_error = f"{type(exc).__name__}: {exc}"
            log(f"[{example.example_id}] solver exception: {rec.solver_error}")
            return rec

        if x_image is None:
            rec.solver_error = "solver returned no image"
            log(f"[{example.example_id}] solver produced no image")
            return rec

        # Persist the solver output once; every evaluator is reading the same X.
        img_path = os.path.join(out_dir, f"x_{example.example_id}.png")
        x_image.save(img_path)
        rec.solver_image_path = os.path.basename(img_path)

    # Verifier (against held-out D) and all validators (without D) can run
    # in parallel — they all share the same x_image input.
    def _do_verify() -> tuple[Optional[bool], str, Optional[str]]:
        if verifier is None:
            return None, "", None
        try:
            vr = verifier.run(x_image, reference_d)
            return vr.is_same_chemical, vr.explanation, None
        except Exception as exc:
            tb = traceback.format_exc()
            return None, "", f"{type(exc).__name__}: {exc}\n{tb}"

    def _do_validate(v: Validator) -> ValidatorVerdict:
        try:
            vr = v.run(image_a, image_b, image_c, x_image)
            return ValidatorVerdict(
                model=v.model, is_valid=vr.is_valid_reactant,
                explanation=vr.explanation,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            return ValidatorVerdict(
                model=v.model, is_valid=None, explanation="",
                error=f"{type(exc).__name__}: {exc}\n{tb}",
            )

    # Cap intra-example concurrency. Fanning out 13+ evaluator calls
    # simultaneously caused providers to queue requests and hit the 300s
    # wall-clock cap; capping the pool at e.g. 4 keeps the burst small
    # enough to flow through.
    inner_workers = max(1, evaluators_per_example)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=inner_workers, thread_name_prefix="eval"
    ) as pool:
        ver_fut = pool.submit(_do_verify)
        val_futs = {pool.submit(_do_validate, v): v.model for v in validators}
        is_correct, ver_exp, ver_err = ver_fut.result()
        verdicts = [f.result() for f in concurrent.futures.as_completed(val_futs)]

    rec.verifier_correct = is_correct
    rec.verifier_explanation = ver_err or ver_exp
    # Preserve user-given validator order in the output rather than completion order.
    by_model = {vd.model: vd for vd in verdicts}
    rec.validator_verdicts = [by_model[v.model] for v in validators]
    log(f"[{example.example_id}] ✓ verifier={is_correct} "
        f"validators={[(vd.model.split('/')[-1], vd.is_valid) for vd in rec.validator_verdicts]}")
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AgreementStats:
    """Per-validator confusion matrix vs verifier (ground truth) plus
    discrepancy-focused statistics.

    Class convention: positive = "solver was correct" (verifier said yes).
      TP: validator said valid AND verifier said correct
      FP: validator said valid AND verifier said wrong   (over-permissive)
      TN: validator said invalid AND verifier said wrong
      FN: validator said invalid AND verifier said correct (over-strict)
    """
    model: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    errored: int

    @property
    def agreement(self) -> float:
        d = self.tp + self.fp + self.tn + self.fn
        return 100.0 * (self.tp + self.tn) / d if d else 0.0

    @property
    def disagreement(self) -> float:
        d = self.tp + self.fp + self.tn + self.fn
        return 100.0 * (self.fp + self.fn) / d if d else 0.0

    @property
    def sensitivity(self) -> float:  # P(valid | correct)
        d = self.tp + self.fn
        return 100.0 * self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:  # P(invalid | wrong)
        d = self.tn + self.fp
        return 100.0 * self.tn / d if d else 0.0

    @property
    def fp_rate(self) -> float:
        """Over-permissive: of the wrong reactants, what fraction did the
        validator approve? 1 - specificity."""
        d = self.fp + self.tn
        return 100.0 * self.fp / d if d else 0.0

    @property
    def fn_rate(self) -> float:
        """Over-strict: of the correct reactants, what fraction did the
        validator reject? 1 - sensitivity."""
        d = self.fn + self.tp
        return 100.0 * self.fn / d if d else 0.0

    @property
    def cohens_kappa(self) -> float:
        """Chance-adjusted agreement on binary outcomes; 0=chance, 1=perfect."""
        n = self.tp + self.fp + self.tn + self.fn
        if n == 0:
            return 0.0
        po = (self.tp + self.tn) / n
        # marginals
        p_validator_yes = (self.tp + self.fp) / n
        p_verifier_yes  = (self.tp + self.fn) / n
        pe = p_validator_yes * p_verifier_yes + (1 - p_validator_yes) * (1 - p_verifier_yes)
        return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0

    @property
    def mcc(self) -> float:
        """Matthews correlation; for binary data this equals the phi coefficient,
        which is the Pearson correlation between the two binary judgments. So
        this is the "covariance-style" relatedness measure. Range [-1, 1]."""
        import math
        denom_sq = (self.tp + self.fp) * (self.tp + self.fn) * \
                   (self.tn + self.fp) * (self.tn + self.fn)
        if denom_sq == 0:
            return 0.0
        return (self.tp * self.tn - self.fp * self.fn) / math.sqrt(denom_sq)

    @property
    def mcnemar_p(self) -> float:
        """p-value for McNemar's test of marginal homogeneity.

        H0: P(validator says valid) = P(verifier says correct).
        Asks whether the validator is *systematically biased* (FP ≠ FN beyond
        chance), not whether it merely disagrees at random.

        Small N (FP+FN < 25): exact two-sided binomial test on min(FP, FN)
        with p=0.5. Larger: continuity-corrected χ² approximation.
        """
        import math
        b = self.fp + self.fn
        if b == 0:
            return 1.0  # nothing discordant; can't reject H0
        if b < 25:
            k = min(self.fp, self.fn)
            tail = sum(math.comb(b, i) for i in range(k + 1)) * (0.5 ** b)
            return min(1.0, 2.0 * tail)
        # χ²(1) with continuity correction; p = erfc(sqrt(chi2/2))
        chi2 = (abs(self.fp - self.fn) - 1) ** 2 / b
        return math.erfc(math.sqrt(chi2 / 2))

    @property
    def kappa_p(self) -> float:
        """p-value for Cohen's κ under H0: κ = 0 (asymptotic, two-sided)."""
        import math
        n = self.tp + self.fp + self.tn + self.fn
        if n == 0:
            return 1.0
        p_v_yes = (self.tp + self.fp) / n   # validator marginal
        p_v_no  = 1 - p_v_yes
        p_t_yes = (self.tp + self.fn) / n   # verifier marginal
        p_t_no  = 1 - p_t_yes
        pe = p_v_yes * p_t_yes + p_v_no * p_t_no
        if abs(1 - pe) < 1e-12:
            return 1.0
        # Standard formula for SE of kappa under H0 (Fleiss, 1981) for 2x2 tables.
        sum_term = (p_v_yes * p_t_yes * (p_v_yes + p_t_yes)
                    + p_v_no * p_t_no * (p_v_no + p_t_no))
        var0 = (pe + pe * pe - sum_term) / (n * (1 - pe) ** 2)
        if var0 <= 0:
            return 1.0
        z = self.cohens_kappa / math.sqrt(var0)
        # Two-sided normal p-value: p = erfc(|z|/sqrt(2))
        return math.erfc(abs(z) / math.sqrt(2))


def _compute_agreement(records: list[ExampleRecord], validators: list[str]) -> list[AgreementStats]:
    out: list[AgreementStats] = []
    for vm in validators:
        tp = fp = tn = fn = errored = n = 0
        for rec in records:
            if rec.solver_error or rec.verifier_correct is None:
                continue
            vd = next((v for v in rec.validator_verdicts if v.model == vm), None)
            if vd is None or vd.error or vd.is_valid is None:
                errored += 1
                continue
            n += 1
            if vd.is_valid and rec.verifier_correct:     tp += 1
            elif vd.is_valid and not rec.verifier_correct: fp += 1
            elif not vd.is_valid and not rec.verifier_correct: tn += 1
            else: fn += 1
        out.append(AgreementStats(
            model=vm, n=n, tp=tp, fp=fp, tn=tn, fn=fn, errored=errored,
        ))
    return out


def _pairwise_agreement(records: list[ExampleRecord], validators: list[str]) -> dict:
    """For each pair of validators, fraction of examples where they agreed."""
    out = {}
    for i, a in enumerate(validators):
        for b in validators[i + 1 :]:
            agree = total = 0
            for rec in records:
                if rec.solver_error: continue
                va = next((v for v in rec.validator_verdicts if v.model == a), None)
                vb = next((v for v in rec.validator_verdicts if v.model == b), None)
                if not va or not vb or va.is_valid is None or vb.is_valid is None:
                    continue
                total += 1
                if va.is_valid == vb.is_valid:
                    agree += 1
            out[f"{a} vs {b}"] = (agree, total)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_csv(out_dir: str, records: list[ExampleRecord], validators: list[str]) -> str:
    path = os.path.join(out_dir, "decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "difficulty", "solver_error", "verifier_correct", *validators])
        for r in records:
            row = [r.example_id, r.difficulty or "", r.solver_error or "",
                   "" if r.verifier_correct is None else r.verifier_correct]
            for vm in validators:
                vd = next((v for v in r.validator_verdicts if v.model == vm), None)
                if vd is None or vd.is_valid is None:
                    row.append("err" if (vd and vd.error) else "")
                else:
                    row.append(vd.is_valid)
            w.writerow(row)
    return path


def _write_results_json(out_dir: str, records: list[ExampleRecord], metadata: dict) -> str:
    data = []
    for r in records:
        data.append({
            "example_id": r.example_id,
            "difficulty": r.difficulty,
            "input_paths": r.input_paths,
            "solver_image_path": r.solver_image_path,
            "solver_error": r.solver_error,
            "verifier_correct": r.verifier_correct,
            "verifier_explanation": r.verifier_explanation,
            "validator_verdicts": [
                {"model": v.model, "is_valid": v.is_valid,
                 "explanation": v.explanation, "error": v.error}
                for v in r.validator_verdicts
            ],
        })
    path = os.path.join(out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "records": data}, f, indent=2)
    return path


def _format_summary(
    records: list[ExampleRecord],
    stats: list[AgreementStats],
    pair_stats: dict,
    metadata: dict,
) -> str:
    n_total = len(records)
    n_solver_err = sum(1 for r in records if r.solver_error)
    n_correct = sum(1 for r in records if r.verifier_correct is True)
    n_wrong   = sum(1 for r in records if r.verifier_correct is False)
    lines = [
        f"=== validator comparison — {metadata['timestamp_utc']} ===",
        f"data_dir: {metadata['data_dir']}  examples={n_total}",
        f"solver:   {metadata['solver_model']}  (single shot, no iteration)",
        f"verifier: {metadata['verifier_model']}",
        f"validators ({len(metadata['validator_models'])}):",
        *[f"  - {m}" for m in metadata["validator_models"]],
        "",
        f"solver outcomes:  ok={n_total - n_solver_err}  errored={n_solver_err}",
        f"verifier ground truth (of solver-ok): correct={n_correct}  wrong={n_wrong}",
        "",
        "=== per-validator discrepancy vs verifier (ground truth) ===",
        f"{'model':45s} {'n':>4s} {'disagree%':>10s} {'FP%':>6s} {'FN%':>6s} "
        f"{'kappa':>6s} {'p(K)':>7s} {'MCC=phi':>8s} {'p(McN)':>8s} "
        f"{'sens%':>6s} {'spec%':>6s} "
        f"{'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} {'err':>4s}",
        "-" * 150,
    ]
    for s in stats:
        lines.append(
            f"{s.model:45s} {s.n:>4d} {s.disagreement:>9.2f}% "
            f"{s.fp_rate:>5.2f}% {s.fn_rate:>5.2f}% "
            f"{s.cohens_kappa:>6.3f} {s.kappa_p:>7.4f} "
            f"{s.mcc:>8.3f} {s.mcnemar_p:>8.4f} "
            f"{s.sensitivity:>5.2f}% {s.specificity:>5.2f}% "
            f"{s.tp:>4d} {s.fp:>4d} {s.tn:>4d} {s.fn:>4d} {s.errored:>4d}"
        )
    lines.append("")
    lines.append("legend:")
    lines.append("  disagree% = (FP+FN)/N    — raw rate where validator and verifier differ")
    lines.append("  FP% = FP/(FP+TN)         — over-permissive: % of wrong reactants the validator approved")
    lines.append("  FN% = FN/(FN+TP)         — over-strict:    % of correct reactants the validator rejected")
    lines.append("  kappa  Cohen's           — chance-adjusted agreement (0=chance, 1=perfect)")
    lines.append("  p(K)                     — asymptotic two-sided p-value for H0: kappa=0")
    lines.append("  MCC=phi                  — Matthews correlation; for binary data == phi == Pearson corr [-1,1]")
    lines.append("  p(McN)                   — McNemar p-value; small ⇒ validator is *systematically* biased (FP ≠ FN)")
    if pair_stats:
        lines.append("")
        lines.append("=== pairwise validator agreement ===")
        for pair, (agree, total) in pair_stats.items():
            pct = (100.0 * agree / total) if total else 0.0
            lines.append(f"  {pair}: {agree}/{total} ({pct:.2f}%)")
    return "\n".join(lines)


def _write_html_report(
    out_dir: str,
    records: list[ExampleRecord],
    stats: list[AgreementStats],
    summary_text: str,
    metadata: dict,
) -> str:
    def b64(p: str) -> str:
        if not p or not os.path.exists(p):
            return ""
        with open(p, "rb") as f:
            import base64
            return base64.b64encode(f.read()).decode("ascii")

    def img_tag(path: str | None) -> str:
        if not path:
            return "<i>—</i>"
        full = path if os.path.isabs(path) else os.path.join(out_dir, path)
        b = b64(full)
        if not b:
            return f"<i>missing: {html.escape(path)}</i>"
        return f'<img src="data:image/png;base64,{b}" style="max-width:160px;">'

    def cell(v: ValidatorVerdict | None) -> str:
        if v is None: return "<td>—</td>"
        if v.error:   return f'<td title="{html.escape(v.error)}" style="color:#a00">err</td>'
        if v.is_valid is None: return "<td>—</td>"
        sym = "✓" if v.is_valid else "✗"
        color = "#070" if v.is_valid else "#a00"
        return (f'<td style="text-align:center;color:{color};font-weight:bold;" '
                f'title="{html.escape((v.explanation or "")[:600])}">{sym}</td>')

    rows = []
    for r in records:
        if r.solver_error:
            rows.append(f"<tr><td><code>{html.escape(r.example_id)}</code></td>"
                        f"<td>{html.escape(r.difficulty or '')}</td>"
                        f"<td colspan='{2 + len(metadata['validator_models'])}' style='color:#a00'>"
                        f"solver error: {html.escape(r.solver_error)}</td></tr>")
            continue
        ver_cell = "—" if r.verifier_correct is None else (
            f'<span style="color:{"#070" if r.verifier_correct else "#a00"};font-weight:bold">'
            f'{"✓" if r.verifier_correct else "✗"}</span>'
        )
        cells = "".join(cell(v) for v in r.validator_verdicts)
        rows.append(
            f"<tr><td><code>{html.escape(r.example_id)}</code></td>"
            f"<td>{html.escape(r.difficulty or '')}</td>"
            f"<td>{img_tag(r.solver_image_path)}</td>"
            f"<td style='text-align:center'>{ver_cell}</td>"
            f"{cells}</tr>"
        )

    val_headers = "".join(f"<th>{html.escape(m.split('/')[-1])}</th>"
                          for m in metadata["validator_models"])
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MOSAIC validator comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; vertical-align: middle; }}
  th {{ background: #f4f4f4; text-align: left; }}
  pre {{ white-space: pre-wrap; }}
</style></head><body>
<h1>MOSAIC — validator comparison</h1>
<pre>{html.escape(summary_text)}</pre>
<h2>Per-example decisions</h2>
<table>
  <thead><tr>
    <th>id</th><th>difficulty</th><th>X (solver)</th><th>verifier (truth)</th>{val_headers}
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</body></html>""")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single solver call per example, then evaluate the same "
                    "image with multiple validators in parallel.",
    )
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    p.add_argument("--solver-model", default=_DEFAULT_SOLVER)
    p.add_argument("--validator-models", nargs="+", default=_DEFAULT_VALIDATORS,
                   help=f"Validator models to compare. Default: "
                        f"{' '.join(_DEFAULT_VALIDATORS)}.  Curated options: "
                        f"{' '.join(VALIDATOR_MODELS_OPENROUTER)}.")
    p.add_argument("--verifier-model", default="gemini-3-flash-preview",
                   help="Single verifier model (uses ground-truth D image). "
                        "Bare name → Gemini direct API.")
    p.add_argument("--no-verifier", action="store_true",
                   help="Skip the verifier; only emit validator decisions (useful "
                        "if you don't want per-example ground truth).")

    p.add_argument("--max-workers", type=int, default=4,
                   help="Concurrent examples (each example runs solver sequentially "
                        "then verifier+all validators in parallel internally).")
    p.add_argument("--evaluators-per-example", type=int, default=4,
                   help="Cap on concurrent verifier+validator calls within a "
                        "single example. Lower this if providers are queueing "
                        "/timing out under burst (default: 4).")

    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "runs"))
    p.add_argument("--api-key-file", default=None,
                   help="Path to a file containing the OpenRouter API key. "
                        "Falls back to $OPENROUTER_API_KEY.")
    p.add_argument("--google-api-key-file", default=None,
                   help="Path to a file containing the Google AI API key. "
                        "Falls back to $GOOGLE_API_KEY.")
    p.add_argument("--openrouter-base-url", default=OPENROUTER_BASE_URL)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--example-ids", nargs="+", default=None,
                   help="If given, restrict to these example_ids (after load).")

    p.add_argument("--reuse-solver-from", default=None,
                   help="Path to a per-model run subdir containing iter_images/. "
                        "When set, the solver is NOT called; the X image for each "
                        "example is loaded from "
                        "<reuse-solver-from>/iter_images/<example_id>/iter_<N>.png "
                        "instead. Skips solver-related API key checks.")
    p.add_argument("--reuse-iter", type=int, default=1,
                   help="Iteration index N to load when --reuse-solver-from is set "
                        "(1 = first attempt; matches MOSAIC's 1-indexed iteration "
                        "filenames).")
    return p.parse_args(argv)


def _resolve_key(
    path: str | None, env_var: str, default_file: str | None = None
) -> str | None:
    """Resolve an API key in precedence order: --flag, env var, default file."""
    if path:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            return f.read().strip() or None
    val = os.environ.get(env_var)
    if val:
        return val
    if default_file and os.path.exists(default_file):
        with open(default_file, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    or_key = _resolve_key(
        args.api_key_file, "OPENROUTER_API_KEY", _DEFAULT_OPENROUTER_KEY_FILE
    )
    g_key = _resolve_key(
        args.google_api_key_file, "GOOGLE_API_KEY", _DEFAULT_GOOGLE_KEY_FILE
    )

    # Sanity-check keys vs the actual route each model will take. In reuse
    # mode we never call the solver, so don't require its key.
    all_models = list(args.validator_models)
    if not args.reuse_solver_from:
        all_models.append(args.solver_model)
    if not args.no_verifier:
        all_models.append(args.verifier_model)
    providers = {pick_provider(m, g_key) for m in all_models}
    if "openrouter" in providers and not or_key:
        print("error: OpenRouter API key missing for one or more models. "
              "Set $OPENROUTER_API_KEY, pass --api-key-file, or place it at "
              ".secrets/openrouter_api_key.", file=sys.stderr)
        return 2
    if "gemini" in providers and not g_key:
        print("error: Gemini API key missing for one or more Gemini models. "
              "Set $GOOGLE_API_KEY, pass --google-api-key-file, or place it at "
              ".secrets/google_api_key.", file=sys.stderr)
        return 2

    examples = load_examples(args.data_dir)
    if not examples:
        print(f"error: no examples in {args.data_dir}", file=sys.stderr); return 1
    if args.example_ids:
        keep = set(args.example_ids)
        examples = [e for e in examples if e.example_id in keep]
        if not examples:
            print("error: --example-ids matched no examples", file=sys.stderr); return 1
    else:
        if args.stride < 1 or args.offset < 0 or args.offset >= max(args.stride, 1):
            print("error: bad --stride/--offset", file=sys.stderr); return 2
        if args.stride > 1 or args.offset > 0:
            examples = examples[args.offset::args.stride]
        if args.limit:
            examples = examples[: args.limit]
    print(f"[run_validators] running on {len(examples)} examples")

    # Backend cache by provider — solver, verifier, and validators may share a backend.
    backend_cache: dict[str, Backend] = {}

    def get_backend(model: str) -> Backend:
        key = pick_provider(model, g_key)
        if key not in backend_cache:
            backend_cache[key] = make_backend(
                model,
                openrouter_api_key=or_key,
                google_api_key=g_key,
                openrouter_base_url=args.openrouter_base_url,
                openrouter_max_concurrent_calls=max(8, args.max_workers * (1 + len(args.validator_models))),
            )
        return backend_cache[key]

    solver = None if args.reuse_solver_from else Solver(
        get_backend(args.solver_model), args.solver_model
    )
    verifier = None if args.no_verifier else Verifier(
        get_backend(args.verifier_model), args.verifier_model, prompts.VERIFIER_PROMPT,
    )
    validators = [
        Validator(get_backend(m), m, prompts.VALIDATOR_PROMPT)
        for m in args.validator_models
    ]

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"validators_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run_validators] artifacts under {out_dir}")
    if args.reuse_solver_from:
        print(f"[run_validators] REUSE mode: loading X images from "
              f"{args.reuse_solver_from}/iter_images/<id>/iter_{args.reuse_iter}.png")
    else:
        print(f"[run_validators] solver={args.solver_model}")
    print(f"[run_validators] verifier={'off' if args.no_verifier else args.verifier_model}")
    print(f"[run_validators] {len(validators)} validators: {', '.join(args.validator_models)}")

    log_lock = threading.Lock()
    def log(msg: str) -> None:
        with log_lock: print(msg, flush=True)

    records: list[ExampleRecord] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers, thread_name_prefix="ex"
    ) as pool:
        futs = {
            pool.submit(_run_one_example,
                        example=e, solver=solver, verifier=verifier,
                        validators=validators, out_dir=out_dir,
                        reuse_solver_from=args.reuse_solver_from,
                        reuse_iter=args.reuse_iter,
                        evaluators_per_example=args.evaluators_per_example,
                        log=log): e
            for e in examples
        }
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="examples"):
            ex = futs[fut]
            try:
                records.append(fut.result())
            except Exception as exc:
                traceback.print_exc()
                records.append(ExampleRecord(
                    example_id=ex.example_id, difficulty=_difficulty_of(ex),
                    input_paths=ex.all_paths, solver_image_path=None,
                    solver_error=f"orchestration: {type(exc).__name__}: {exc}",
                    verifier_correct=None, verifier_explanation="",
                    validator_verdicts=[],
                ))
    records.sort(key=lambda r: r.example_id)

    metadata = {
        "timestamp_utc": timestamp, "data_dir": args.data_dir,
        "solver_model": (
            f"REUSED from {args.reuse_solver_from} iter={args.reuse_iter}"
            if args.reuse_solver_from else args.solver_model
        ),
        "verifier_model": None if args.no_verifier else args.verifier_model,
        "validator_models": list(args.validator_models),
        "n_examples": len(records),
        "reuse_solver_from": args.reuse_solver_from,
        "reuse_iter": args.reuse_iter if args.reuse_solver_from else None,
    }
    stats = _compute_agreement(records, args.validator_models) if not args.no_verifier else []
    pair_stats = _pairwise_agreement(records, args.validator_models)

    summary = _format_summary(records, stats, pair_stats, metadata)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    _write_results_json(out_dir, records, metadata)
    _write_csv(out_dir, records, args.validator_models)
    _write_html_report(out_dir, records, stats, summary, metadata)

    print()
    print(summary)
    print(f"\n[run_validators] artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
