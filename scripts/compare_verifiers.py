"""Compare two verifier models on the same (predicted X, reference D) image pairs.

Reads a MOSAIC eval run directory (one created by ``mosaic.cli``) and, for
each (solver, example_id, iteration) tuple that has a saved ``iter_<n>.png``:

  - finds the predicted X image (``<solver_dir>/iter_images/<example_id>/iter_<n>.png``)
  - finds the reference D image (from ``results.json`` ``input_paths[3]``)
  - reads the OLD verifier verdict already stored in ``results.json``
    (no need to recall the old verifier — we trust the persisted artifact)
  - re-runs the NEW verifier on the (predicted, reference) pair

Then writes:

  <run_dir>/verifier_compare/<safe-old>__vs__<safe-new>/
    comparison.json       per-pair old vs new verdict + new explanation
    comparison.html       all pairs with image previews + both verdicts
    disagreements.html    subset where old.is_same_chemical != new.is_same_chemical

Concurrency is per-pair via ``concurrent.futures.ThreadPoolExecutor``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import html
import json
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import PIL.Image

from mosaic.agents import Verifier, make_backend, pick_provider
from mosaic.models import OPENROUTER_BASE_URL
from mosaic import prompts


def _safe_filename(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


@dataclass
class Pair:
    solver: str
    example_id: str
    iteration: int
    predicted_path: str
    reference_path: str
    old_is_same: bool
    old_explanation: str
    # filled in after the new verifier runs
    new_is_same: Optional[bool] = None
    new_explanation: str = ""
    new_elapsed_s: float = 0.0
    error: Optional[str] = None


def _read_key(path: Optional[str], env_var: str) -> Optional[str]:
    if path:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            return f.read().strip()
    return os.environ.get(env_var)


def _walk_pairs(run_dir: str) -> tuple[list[Pair], dict]:
    """Walk run_dir and produce all (predicted, reference) pairs that exist on disk.

    Returns (pairs, metadata). metadata captures the run-level info (data_dir,
    old_verifier name) for the report header.
    """
    pairs: list[Pair] = []
    metadata = {"old_verifier": "gemini-3-flash-preview", "data_dir": None,
                "validator": None, "solvers": []}
    for entry in sorted(os.listdir(run_dir)):
        solver_dir = os.path.join(run_dir, entry)
        results_path = os.path.join(solver_dir, "results.json")
        if not (os.path.isdir(solver_dir) and os.path.isfile(results_path)):
            continue
        metadata["solvers"].append(entry)
        examples = json.load(open(results_path))
        for ex in examples:
            ex_id = ex["example_id"]
            input_paths = ex.get("input_paths") or []
            if len(input_paths) < 4:
                continue
            reference_path = input_paths[3]
            for it in ex.get("iterations", []):
                if not it.get("solver_produced_image"):
                    continue
                predicted_path = os.path.join(
                    solver_dir, "iter_images", ex_id, f"iter_{it['iteration']}.png"
                )
                if not os.path.isfile(predicted_path):
                    continue
                ver = it.get("verification") or {}
                if "is_same_chemical" not in ver:
                    continue
                pairs.append(Pair(
                    solver=entry,
                    example_id=ex_id,
                    iteration=it["iteration"],
                    predicted_path=predicted_path,
                    reference_path=reference_path,
                    old_is_same=bool(ver["is_same_chemical"]),
                    old_explanation=ver.get("explanation") or "",
                ))
    return pairs, metadata


def _run_new_verifier(verifier: Verifier, pair: Pair) -> Pair:
    t0 = time.time()
    try:
        predicted = PIL.Image.open(pair.predicted_path)
        reference = PIL.Image.open(pair.reference_path)
        result = verifier.run(predicted, reference)
        pair.new_is_same = bool(result.is_same_chemical)
        pair.new_explanation = result.explanation
    except Exception as exc:
        pair.error = f"{type(exc).__name__}: {exc}"
    pair.new_elapsed_s = time.time() - t0
    return pair


def _verdict_cell(value: Optional[bool]) -> str:
    if value is None:
        return '<span style="color:#888;">—</span>'
    return '<span style="color:#0a7a0a;font-weight:600;">✓</span>' if value else '<span style="color:#a10000;font-weight:600;">✗</span>'


def _render_html(out_path: str, pairs: list[Pair], metadata: dict, new_verifier: str, only_disagreements: bool = False) -> None:
    title_suffix = " — disagreements only" if only_disagreements else ""
    rows = []
    n_pairs = 0
    for p in pairs:
        if only_disagreements:
            if p.error or p.new_is_same is None or p.old_is_same == p.new_is_same:
                continue
        n_pairs += 1
        pred_uri = pathlib.Path(p.predicted_path).resolve().as_uri()
        ref_uri = pathlib.Path(p.reference_path).resolve().as_uri()
        new_cell = '<span style="color:#888;">error</span>' if p.error else _verdict_cell(p.new_is_same)
        agree = "—" if p.error or p.new_is_same is None else ("✓ agree" if p.old_is_same == p.new_is_same else "✗ disagree")
        agree_color = "#888"
        if not p.error and p.new_is_same is not None:
            agree_color = "#0a7a0a" if p.old_is_same == p.new_is_same else "#a10000"
        err_html = f'<div style="color:#a10000;">error: {html.escape(p.error)}</div>' if p.error else ""
        rows.append(f"""
        <tr>
          <td><code>{html.escape(p.solver)}</code></td>
          <td><code>{html.escape(p.example_id)}</code></td>
          <td style="text-align:center;">{p.iteration}</td>
          <td><img src="{html.escape(pred_uri)}" style="max-width:240px;"></td>
          <td><img src="{html.escape(ref_uri)}" style="max-width:240px;"></td>
          <td style="text-align:center;font-size:1.3em;">{_verdict_cell(p.old_is_same)}</td>
          <td style="text-align:center;font-size:1.3em;">{new_cell}</td>
          <td style="text-align:center;color:{agree_color};font-weight:600;">{agree}</td>
          <td style="font-size:0.85em;"><details><summary>old</summary><div style="max-width:480px;">{html.escape(p.old_explanation)}</div></details>
              <details><summary>new</summary><div style="max-width:480px;">{html.escape(p.new_explanation)}</div></details>
              {err_html}</td>
        </tr>""")
    style = """
      body { font-family: -apple-system, system-ui, sans-serif; max-width: 1700px; margin: 1em auto; padding: 0 1em; color: #222; }
      table { border-collapse: collapse; width: 100%; }
      th, td { padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }
      th { background: #f4f4f4; text-align: left; position: sticky; top: 0; }
      details summary { cursor: pointer; color: #555; }
      .meta { color: #555; font-size: 0.9em; }
    """
    n_total = len(pairs)
    n_agree = sum(1 for p in pairs if p.error is None and p.new_is_same is not None and p.old_is_same == p.new_is_same)
    n_disagree = sum(1 for p in pairs if p.error is None and p.new_is_same is not None and p.old_is_same != p.new_is_same)
    n_err = sum(1 for p in pairs if p.error)
    header = f"""
    <h1>Verifier comparison{title_suffix}</h1>
    <p class="meta">
      run dir: <code>{html.escape(metadata.get('run_dir',''))}</code> ·
      old verifier: <code>{html.escape(metadata.get('old_verifier','?'))}</code> ·
      new verifier: <code>{html.escape(new_verifier)}</code><br>
      total pairs: {n_total} · agree: {n_agree} ({n_agree/max(n_total,1)*100:.1f}%) ·
      disagree: {n_disagree} ({n_disagree/max(n_total,1)*100:.1f}%) ·
      errors: {n_err}{'' if not only_disagreements else f' · showing {n_pairs} disagreements'}
    </p>
    """
    table_head = """
    <tr>
      <th>solver</th><th>example</th><th>iter</th>
      <th>predicted X</th><th>reference D</th>
      <th>old</th><th>new</th><th>agree</th>
      <th>explanations</th>
    </tr>
    """
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>verifier compare{title_suffix}</title>
<style>{style}</style></head><body>
{header}
<table><thead>{table_head}</thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="A MOSAIC eval run directory.")
    ap.add_argument("--new-verifier", required=True, help="Model name for the new verifier (e.g. 'gemini-3.1-pro-preview' for Gemini direct, 'google/gemini-3.1-pro-preview' for OpenRouter).")
    ap.add_argument("--api-key-file", default=None,
                    help="OpenRouter key file (falls back to $OPENROUTER_API_KEY).")
    ap.add_argument("--google-api-key-file", default=None,
                    help="Gemini-direct key file (falls back to $GOOGLE_API_KEY).")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="Concurrent new-verifier calls (default 8).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only compare the first N pairs (for testing/cost-bounding).")
    ap.add_argument("--solver", action="append", default=None,
                    help="Restrict to specific solver subdir(s); repeatable.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir; default: <run_dir>/verifier_compare/<old>__vs__<new>/")
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir.rstrip("/"))
    if not os.path.isdir(run_dir):
        print(f"error: --run-dir {run_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"[walk] scanning {run_dir}")
    pairs, metadata = _walk_pairs(run_dir)
    metadata["run_dir"] = run_dir
    if args.solver:
        keep = set(args.solver)
        pairs = [p for p in pairs if p.solver in keep]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("error: no pairs found", file=sys.stderr)
        return 1
    print(f"[walk] {len(pairs)} pairs across {len(set(p.solver for p in pairs))} solvers")

    # Backend selection: same gen_config, no overrides — let new verifier run "as-is".
    openrouter_key = _read_key(args.api_key_file, "OPENROUTER_API_KEY")
    google_key = _read_key(args.google_api_key_file, "GOOGLE_API_KEY")
    new_model = args.new_verifier
    provider = pick_provider(new_model, google_key)
    if provider == "gemini" and not google_key:
        print(f"error: {new_model!r} routes to Gemini direct; need $GOOGLE_API_KEY or --google-api-key-file", file=sys.stderr)
        return 2
    if provider == "openrouter" and not openrouter_key:
        print(f"error: {new_model!r} routes to OpenRouter; need $OPENROUTER_API_KEY or --api-key-file", file=sys.stderr)
        return 2
    print(f"[backend] {new_model!r} routing to {provider}")

    backend = make_backend(
        new_model,
        openrouter_api_key=openrouter_key,
        google_api_key=google_key,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_max_concurrent_calls=max(args.max_workers, 4),
        gen_config=None,
    )
    new_verifier = Verifier(backend, new_model, prompts.VERIFIER_PROMPT)

    out_dir = args.out_dir or os.path.join(
        run_dir, "verifier_compare",
        f"{_safe_filename(metadata['old_verifier'])}__vs__{_safe_filename(new_model)}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    progress = {"done": 0}
    progress_lock = threading.Lock()
    t_start = time.time()

    def worker(p: Pair) -> Pair:
        out = _run_new_verifier(new_verifier, p)
        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
            if done % 10 == 0 or done == len(pairs):
                rate = done / max(time.time() - t_start, 0.01)
                eta_s = (len(pairs) - done) / max(rate, 0.001)
                print(f"  [{done}/{len(pairs)}] {rate:.2f}/s, ETA {eta_s/60:.1f} min", flush=True)
        return out

    print(f"[run] {len(pairs)} pairs, {args.max_workers} workers")
    completed: list[Pair] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers, thread_name_prefix="cmp") as ex:
        for fut in concurrent.futures.as_completed([ex.submit(worker, p) for p in pairs]):
            completed.append(fut.result())

    # restore original order (solver, example_id, iteration)
    completed.sort(key=lambda p: (p.solver, p.example_id, p.iteration))

    # write JSON
    json_path = os.path.join(out_dir, "comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {**metadata, "new_verifier": new_model, "n_pairs": len(completed)},
            "pairs": [
                {
                    "solver": p.solver, "example_id": p.example_id, "iteration": p.iteration,
                    "predicted_path": p.predicted_path, "reference_path": p.reference_path,
                    "old_is_same": p.old_is_same, "old_explanation": p.old_explanation,
                    "new_is_same": p.new_is_same, "new_explanation": p.new_explanation,
                    "new_elapsed_s": p.new_elapsed_s, "error": p.error,
                } for p in completed
            ],
        }, f, indent=2)
    print(f"[out] wrote {json_path}")

    # confusion matrix
    n = len(completed)
    n_err = sum(1 for p in completed if p.error)
    n_valid = n - n_err
    yy = sum(1 for p in completed if p.error is None and p.old_is_same and p.new_is_same)
    yn = sum(1 for p in completed if p.error is None and p.old_is_same and not p.new_is_same)
    ny = sum(1 for p in completed if p.error is None and not p.old_is_same and p.new_is_same)
    nn = sum(1 for p in completed if p.error is None and not p.old_is_same and not p.new_is_same)
    n_agree = yy + nn
    n_disagree = yn + ny
    print()
    print(f"=== confusion matrix (old vs new), n={n_valid} pairs (errors={n_err}) ===")
    print(f"                       new=YES   new=NO")
    print(f"  old=YES (same)        {yy:>4}    {yn:>4}")
    print(f"  old=NO  (different)   {ny:>4}    {nn:>4}")
    print(f"  agree: {n_agree}/{n_valid} ({n_agree/max(n_valid,1)*100:.1f}%); disagree: {n_disagree}/{n_valid} ({n_disagree/max(n_valid,1)*100:.1f}%)")

    # write HTML reports
    full_html = os.path.join(out_dir, "comparison.html")
    _render_html(full_html, completed, metadata, new_model, only_disagreements=False)
    print(f"[out] wrote {full_html}")

    disagree_html = os.path.join(out_dir, "disagreements.html")
    _render_html(disagree_html, completed, metadata, new_model, only_disagreements=True)
    print(f"[out] wrote {disagree_html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
