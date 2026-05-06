"""Few-shot ablation: run shots ∈ {1, 3, 5, 7} on the eval set with a small
held-out pool used as the few-shot source.

Defaults (per Sergey's spec, 2026-05-05):
  - solver:    gemini-3-pro-image-preview     (Gemini direct)
  - validator: gemini-3-flash-preview         (Gemini direct)
  - verifier:  gemini-3-flash-preview         (Gemini direct)
  - held-out:  10 examples (stratified, evenly-spaced over sorted IDs)
  - eval set:  remaining 229 of release_data/eval
  - max iterations: 8
  - workers:   12

All Gemini calls go through the Google API key (.secrets/google_api_key)
because the model names are bare ("gemini-..." with no "google/" prefix).

Each shot count gets its own run_eval.py subprocess and lands in
runs/few_shots_<ts>/shots_<n>/run_<inner_ts>/.

Use --smoke to verify wiring end-to-end before the full sweep:
  shots=[1], limit=2, max-iterations=2.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic.data import load_examples  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUN_EVAL = os.path.join(_REPO_ROOT, "run_eval.py")
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "release_data", "eval")

# Bare names → Gemini direct (uses GOOGLE_API_KEY / .secrets/google_api_key)
_DEFAULT_SOLVER = "gemini-3-pro-image-preview"
_DEFAULT_VALIDATOR = "gemini-3-flash-preview"
_DEFAULT_VERIFIER = "gemini-3-flash-preview"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 7],
                   help="Shot counts to sweep. Default: 1 3 5 7")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    p.add_argument("--n-held-out", type=int, default=10,
                   help="Examples held out as the few-shot pool. Stratified by "
                        "evenly-spaced indices over the sorted example list. "
                        "The remaining examples are the eval set. Default: 10.")
    p.add_argument("--few-shot-dir", default=None,
                   help="Override the auto-built few-shot pool. If set, "
                        "--n-held-out is ignored and --data-dir is used as-is.")
    p.add_argument("--solver-model", default=_DEFAULT_SOLVER)
    p.add_argument("--validator-model", default=_DEFAULT_VALIDATOR)
    p.add_argument("--verifier-model", default=_DEFAULT_VERIFIER)
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--max-model-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap examples per run (smoke tests).")
    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "runs"))
    p.add_argument("--gen-config", default=None,
                   help="Pass through to run_eval.py (default: its built-in default).")
    p.add_argument("--smoke", action="store_true",
                   help="Wiring check: shots=[1], limit=2, max-iterations=2.")
    return p.parse_args(argv)


def _find_inner_run_dir(parent: str) -> str | None:
    if not os.path.isdir(parent):
        return None
    subs = sorted(d for d in os.listdir(parent) if d.startswith("run_"))
    return os.path.join(parent, subs[-1]) if subs else None


def _symlink_examples(examples, root_dir: str) -> None:
    """Materialize a dataset dir of real per-example subdirs with symlinked
    images. Required because os.walk doesn't follow directory symlinks."""
    os.makedirs(root_dir, exist_ok=True)
    for e in examples:
        ex_dir = os.path.join(root_dir, e.example_id)
        os.makedirs(ex_dir, exist_ok=True)
        for src in e.all_paths:
            dst = os.path.join(ex_dir, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)


def split_eval_and_pool(
    data_dir: str,
    n_held_out: int,
    eval_dir: str,
    pool_dir: str,
) -> tuple[int, int]:
    """Split data_dir into (eval_set, few_shot_pool) by stratified sampling.

    The held-out pool takes n_held_out examples at evenly-spaced indices over
    the sorted-by-example-id list. Everything else becomes the eval set.
    Both are materialized as symlink dirs so run_eval.py can ingest them.
    """
    examples = sorted(load_examples(data_dir), key=lambda e: e.example_id)
    if n_held_out < 1 or n_held_out >= len(examples):
        raise ValueError(
            f"--n-held-out={n_held_out} out of range for {len(examples)} examples"
        )
    # Evenly-spaced indices to spread the pool across difficulty bands
    # (release_data/eval mixes easy/medium/hard sorted by global ID).
    step = len(examples) / n_held_out
    held_idx = sorted({int(i * step) for i in range(n_held_out)})
    held_set = {examples[i].example_id for i in held_idx}
    pool = [examples[i] for i in held_idx]
    eval_set = [e for e in examples if e.example_id not in held_set]

    _symlink_examples(eval_set, eval_dir)
    _symlink_examples(pool, pool_dir)
    return len(eval_set), len(pool)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.shots = [1]
        args.limit = 2
        args.max_iterations = 2
        print("[few_shots_eval] SMOKE MODE — shots=[1] limit=2 max-iter=2")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if args.smoke else "few_shots"
    parent = os.path.join(args.output_dir, f"{tag}_{ts}")
    os.makedirs(parent, exist_ok=True)
    print(f"[few_shots_eval] parent: {parent}")
    print(f"[few_shots_eval] shots={args.shots}  max_iter={args.max_iterations}  "
          f"workers={args.max_workers}")
    print(f"[few_shots_eval] solver={args.solver_model}")
    print(f"[few_shots_eval] validator={args.validator_model}")
    print(f"[few_shots_eval] verifier={args.verifier_model}")

    if args.few_shot_dir is None:
        eval_dir = os.path.join(parent, "_eval_set")
        few_shot_dir = os.path.join(parent, "_few_shot_pool")
        n_eval, n_pool = split_eval_and_pool(
            args.data_dir, args.n_held_out, eval_dir, few_shot_dir
        )
        print(f"[few_shots_eval] eval set: {n_eval} examples → {eval_dir}")
        print(f"[few_shots_eval] few-shot pool: {n_pool} examples → {few_shot_dir}")
    else:
        eval_dir = args.data_dir
        few_shot_dir = args.few_shot_dir
        print(f"[few_shots_eval] eval set: {eval_dir} (no held-out split)")
        print(f"[few_shots_eval] few-shot pool override: {few_shot_dir}")

    max_n = max(args.shots)
    if args.few_shot_dir is None and max_n > args.n_held_out:
        print(f"error: max shots={max_n} > --n-held-out={args.n_held_out}",
              file=sys.stderr)
        return 2

    rows: list[tuple[int, str, int]] = []
    for n in args.shots:
        sub_out = os.path.join(parent, f"shots_{n}")
        os.makedirs(sub_out, exist_ok=True)
        cmd = [
            sys.executable, _RUN_EVAL,
            "--data-dir", eval_dir,
            "--solver-models", args.solver_model,
            "--validator-model", args.validator_model,
            "--verifier-model", args.verifier_model,
            "--n-few-shots", str(n),
            "--few-shot-dir", few_shot_dir,
            "--max-iterations", str(args.max_iterations),
            "--max-workers", str(args.max_workers),
            "--max-model-workers", str(args.max_model_workers),
            "--output-dir", sub_out,
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.gen_config is not None:
            cmd += ["--gen-config", args.gen_config]

        print(f"\n[few_shots_eval] === shots={n} ===")
        print(f"[few_shots_eval] $ {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        rows.append((n, sub_out, rc))
        print(f"[few_shots_eval] shots={n} returned rc={rc}")

    print("\n" + "=" * 80)
    print(f"[few_shots_eval] sweep complete — parent: {parent}")
    print(f"{'shots':>6s}  {'rc':>3s}  run_dir")
    print("-" * 80)
    for n, out, rc in rows:
        inner = _find_inner_run_dir(out) or out
        print(f"{n:>6d}  {rc:>3d}  {inner}")
        sp = os.path.join(inner, "summary.txt")
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                tail = f.read().splitlines()
            for line in tail[-4:]:
                print(f"        {line}")

    return 0 if all(rc == 0 for _, _, rc in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
