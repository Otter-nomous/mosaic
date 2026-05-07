"""Run one or more solver models against an entire dataset (default: release_data/eval).

Two levels of parallelism:
  - inner: --max-workers examples per model (Pipeline.run_dataset)
  - outer: --max-model-workers solver models running concurrently

Each solver model gets:
  - its own OpenRouterBackend (sized so the backend's wall-clock-cap executor
    doesn't bottleneck inner parallelism)
  - its own Pipeline
  - its own report subdirectory under runs/run_<ts>/<safe-model-name>/

Caching enabled (OpenRouter; no-op on Gemini backend):
  - solver static prefix (prompt + few-shots) — cross-example
  - solver contents[-1]                       — cross-iteration
  - validator prompt + A + B + C              — cross-iteration

Top-level summary.txt aggregates per-model metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import threading
import traceback

import PIL.Image

from mosaic import prompts
from mosaic.agents import (
    Backend,
    Solver,
    Validator,
    Verifier,
    is_gemini_direct,
    make_backend,
)
from mosaic.data import load_examples
from mosaic.metrics import Metrics, compute as compute_metrics, format_summary
from mosaic.models import (
    DEFAULT_VALIDATOR_OPENROUTER,
    DEFAULT_VERIFIER_OPENROUTER,
    OPENROUTER_BASE_URL,
)
from mosaic.pipeline import Pipeline, PipelineConfig, load_few_shots
from mosaic.reporting import ReportPaths, write_report, write_results_json
from mosaic.schemas import IterationRecord, ValidationResult, VerificationResult


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "release_data", "eval")
_DEFAULT_OPENROUTER_KEY_FILE = os.path.join(_REPO_ROOT, ".secrets", "openrouter_api_key")
_DEFAULT_GOOGLE_KEY_FILE = os.path.join(_REPO_ROOT, ".secrets", "google_api_key")
_DEFAULT_GEN_CONFIG = os.path.join(_REPO_ROOT, "gen_config_v3.json")
_DEFAULT_SOLVERS = [
    "openai/gpt-5.4-image-2",
    "google/gemini-3.1-flash-image-preview",
]


def _safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _load_prior_iterations(
    solver_subdir: str,
) -> dict[str, list[IterationRecord]]:
    """Reconstruct per-example iteration history from a prior run's solver dir.

    Reads ``results.json`` for the structured fields (validation/verification),
    then loads each iteration's solver image from ``iter_images/<id>/iter_N.png``
    if present. Iterations whose original solver call returned no image are
    reconstructed with ``solver_output_image=None`` (the correction-block
    builder already handles this).
    """
    results_path = os.path.join(solver_subdir, "results.json")
    images_root = os.path.join(solver_subdir, "iter_images")
    with open(results_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    out: dict[str, list[IterationRecord]] = {}
    for r in rows:
        if r.get("error"):
            # Pipeline crashed before any iterations — start over from scratch.
            continue
        ex_id = r["example_id"]
        ex_img_dir = os.path.join(images_root, ex_id)
        records: list[IterationRecord] = []
        for it in r.get("iterations", []):
            n = it["iteration"]
            img: PIL.Image.Image | None = None
            if it.get("solver_produced_image"):
                p = os.path.join(ex_img_dir, f"iter_{n}.png")
                if os.path.exists(p):
                    img = PIL.Image.open(p)
                    img.load()  # eager-load so the file handle can close
            records.append(IterationRecord(
                iteration=n,
                solver_output_image=img,
                solver_prompt_text=f"<resumed from prior run; iter {n} prompt not preserved>",
                validation=ValidationResult(**it["validation"]),
                verification=VerificationResult(**it["verification"]),
            ))
        if records:
            out[ex_id] = records
    return out


def _persist_iteration_images(sub_dir: str, results: list) -> None:
    """Write each iteration's solver_output_image to iter_images/<id>/iter_N.png.

    Idempotent: overwrites existing files. Enables future ``--resume-from`` runs.
    """
    img_root = os.path.join(sub_dir, "iter_images")
    for r in results:
        if r.error:
            continue
        for it in r.iterations:
            if it.solver_output_image is None:
                continue
            ex_dir = os.path.join(img_root, r.example_id)
            os.makedirs(ex_dir, exist_ok=True)
            it.solver_output_image.save(
                os.path.join(ex_dir, f"iter_{it.iteration}.png"), format="PNG"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run solver model(s) on a full MOSAIC dataset (eval by default).",
    )
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                   help=f"Dataset root (recursive). Default: {_DEFAULT_DATA_DIR}")
    p.add_argument("--solver-models", nargs="+", default=_DEFAULT_SOLVERS,
                   help=f"OpenRouter solver model(s). Default: {' '.join(_DEFAULT_SOLVERS)}")
    p.add_argument("--validator-model", default=DEFAULT_VALIDATOR_OPENROUTER)
    p.add_argument("--verifier-model", default=DEFAULT_VERIFIER_OPENROUTER)
    p.add_argument("--no-verifier", action="store_true",
                   help="Disable verifier (no D-image comparison).")

    p.add_argument("--few-shot-dir", default=None,
                   help="Optional separate dataset directory to draw few-shots from. "
                        "Eval set has no train split, so few-shots default to none.")
    p.add_argument("--n-few-shots", type=int, default=0)

    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--max-workers", type=int, default=8,
                   help="Inner parallelism: concurrent examples per model.")
    p.add_argument("--max-model-workers", type=int, default=None,
                   help="Outer parallelism: solver models run concurrently. "
                        "Default: len(--solver-models).")

    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "runs"))
    p.add_argument("--api-key-file", default=None,
                   help="Path to a file containing the OpenRouter API key. "
                        "Resolution order: this flag, then $OPENROUTER_API_KEY, "
                        "then .secrets/openrouter_api_key. Required only if any "
                        "model routes through OpenRouter (slash in name).")
    p.add_argument("--google-api-key-file", default=None,
                   help="Path to a file containing the Gemini (Google AI) API key. "
                        "Resolution order: this flag, then $GOOGLE_API_KEY, then "
                        ".secrets/google_api_key. Required only if any model routes "
                        "to Gemini direct (bare 'gemini-...' name).")
    p.add_argument("--openrouter-base-url", default=OPENROUTER_BASE_URL)
    p.add_argument("--resume-from", default=None,
                   help="Path to a prior runs/run_<ts>/ directory. For each solver, "
                        "examples that already validated are kept as-is; examples that "
                        "didn't validate are continued from iter N+1 up to "
                        "--max-iterations. Requires per-iteration images (auto-created "
                        "by current run_eval.py; for older runs use extract_iter_images.py).")
    p.add_argument("--gen-config", default=_DEFAULT_GEN_CONFIG,
                   help="Path to a JSON file with sampling params (temperature, "
                        "seed, top_p, top_k). Shape: {\"default\": {...}, "
                        "\"per_model\": {model: {...}}}. A null value in per_model "
                        "opts that key out of the default. Applies to BOTH backends. "
                        f"Default: {os.path.relpath(_DEFAULT_GEN_CONFIG, _REPO_ROOT)} "
                        "(if it exists). Pass empty string to disable.")

    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of examples (for smoke tests).")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth example after sorting (e.g. --stride 2 with "
                        "--offset 0 picks 1st, 3rd, 5th, ...).")
    p.add_argument("--offset", type=int, default=0,
                   help="0-indexed starting offset to use with --stride.")
    return p.parse_args(argv)


def _resolve_api_key(
    key_file: str | None, env_var: str, default_file: str | None = None
) -> str | None:
    """Resolve an API key in precedence order: --flag, env var, default file."""
    if key_file:
        with open(os.path.expanduser(key_file), "r", encoding="utf-8") as f:
            return f.read().strip() or None
    val = os.environ.get(env_var)
    if val:
        return val
    if default_file and os.path.exists(default_file):
        with open(default_file, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    return None


def _run_one_model(
    *,
    solver_model: str,
    examples,
    few_shots,
    args: argparse.Namespace,
    openrouter_api_key: str | None,
    google_api_key: str | None,
    gen_config: dict | None,
    prior_iterations_by_id: dict[str, list[IterationRecord]] | None,
    run_dir: str,
    log,
) -> tuple[str, Metrics | None, str | None, str | None]:
    """Run one solver model end-to-end. Returns (model, metrics, report_path, error)."""
    sub_dir = os.path.join(run_dir, _safe_name(solver_model))
    try:
        # Per-model, per-provider backend cache. Solver/validator/verifier may
        # route to different providers (e.g. Gemini-direct solver, OpenRouter
        # validator), but agents on the same provider should share a backend so
        # they share the wall-clock-cap executor and httpx connection pool.
        backend_cache: dict[str, Backend] = {}

        def get_backend(model: str) -> Backend:
            key = "gemini" if is_gemini_direct(model) else "openrouter"
            if key not in backend_cache:
                backend_cache[key] = make_backend(
                    model,
                    openrouter_api_key=openrouter_api_key,
                    google_api_key=google_api_key,
                    openrouter_base_url=args.openrouter_base_url,
                    openrouter_max_concurrent_calls=max(4, args.max_workers * 2),
                    gen_config=gen_config,
                )
            return backend_cache[key]

        config = PipelineConfig(
            solver_prompt=prompts.SOLVER_PROMPT,
            validator_prompt=prompts.VALIDATOR_PROMPT,
            verifier_prompt=None if args.no_verifier else prompts.VERIFIER_PROMPT,
            max_iterations=args.max_iterations,
            max_workers=args.max_workers,
        )
        pipeline = Pipeline(
            solver=Solver(get_backend(solver_model), solver_model),
            validator=Validator(
                get_backend(args.validator_model),
                args.validator_model,
                prompts.VALIDATOR_PROMPT,
            ),
            verifier=None if args.no_verifier else Verifier(
                get_backend(args.verifier_model),
                args.verifier_model,
                prompts.VERIFIER_PROMPT,
            ),
            config=config,
            few_shots=few_shots,
            prior_iterations_by_id=prior_iterations_by_id,
        )
        n_resumed = sum(
            1 for ex in examples if (prior_iterations_by_id or {}).get(ex.example_id)
        )
        log(f"[{solver_model}] starting on {len(examples)} examples "
            f"({n_resumed} resumed from prior run)")
        results = pipeline.run_dataset(examples)
        metrics = compute_metrics(results)

        paths = ReportPaths.make(sub_dir)
        _persist_iteration_images(sub_dir, results)
        prompt_map = {
            "Solver": config.solver_prompt,
            "Validator": config.validator_prompt,
            "Verifier": config.verifier_prompt or "",
        }
        write_report(paths, results, metrics, prompt_map)
        write_results_json(paths, results)
        log(f"[{solver_model}] done — acc={metrics.final_accuracy:.2f}% "
            f"valid={metrics.final_validity:.2f}% — {paths.main}")
        return solver_model, metrics, paths.main, None
    except Exception as exc:
        traceback.print_exc()
        return solver_model, None, None, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    openrouter_api_key = _resolve_api_key(
        args.api_key_file, "OPENROUTER_API_KEY", _DEFAULT_OPENROUTER_KEY_FILE
    )
    google_api_key = _resolve_api_key(
        args.google_api_key_file, "GOOGLE_API_KEY", _DEFAULT_GOOGLE_KEY_FILE
    )

    # Each model routes to a backend by name; only the keys actually needed
    # have to be present. Compute who needs what so we can fail fast.
    all_models: list[str] = list(args.solver_models) + [args.validator_model]
    if not args.no_verifier:
        all_models.append(args.verifier_model)
    needs_openrouter = any(not is_gemini_direct(m) for m in all_models)
    needs_google = any(is_gemini_direct(m) for m in all_models)
    if needs_openrouter and not openrouter_api_key:
        print("error: OpenRouter API key missing for one or more models. "
              "Set $OPENROUTER_API_KEY or pass --api-key-file.", file=sys.stderr)
        return 2
    if needs_google and not google_api_key:
        print("error: Gemini API key missing for one or more bare 'gemini-...' "
              "models. Set $GOOGLE_API_KEY or pass --google-api-key-file.",
              file=sys.stderr)
        return 2

    gen_config: dict | None = None
    if args.gen_config:
        path = os.path.expanduser(args.gen_config)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                gen_config = json.load(f)
            print(f"[run_eval] gen_config loaded from {path}: "
                  f"default={gen_config.get('default')}")
        elif args.gen_config != _DEFAULT_GEN_CONFIG:
            print(f"error: --gen-config {args.gen_config!r} not found.", file=sys.stderr)
            return 2

    prior_by_solver: dict[str, dict[str, list[IterationRecord]]] = {}
    if args.resume_from:
        resume_root = os.path.expanduser(args.resume_from)
        if not os.path.isdir(resume_root):
            print(f"error: --resume-from {resume_root!r} is not a directory.",
                  file=sys.stderr)
            return 2
        for m in args.solver_models:
            sub = os.path.join(resume_root, _safe_name(m))
            res = os.path.join(sub, "results.json")
            if not os.path.exists(res):
                print(f"[run_eval] resume: no prior data for {m!r} "
                      f"(expected {res}); will run from scratch")
                continue
            prior_by_solver[m] = _load_prior_iterations(sub)
            n_done = sum(
                1 for its in prior_by_solver[m].values() if its and its[-1].validation.is_valid_reactant
            )
            print(f"[run_eval] resume {m}: loaded {len(prior_by_solver[m])} prior examples "
                  f"({n_done} already validated, {len(prior_by_solver[m]) - n_done} to continue)")

    examples = load_examples(args.data_dir)
    if not examples:
        print(f"error: no examples found in {args.data_dir}", file=sys.stderr)
        return 1
    if args.stride < 1 or args.offset < 0 or args.offset >= max(args.stride, 1):
        print(f"error: invalid --stride {args.stride} / --offset {args.offset}; "
              f"need stride>=1 and 0<=offset<stride.", file=sys.stderr)
        return 2
    if args.stride > 1 or args.offset > 0:
        n_before = len(examples)
        examples = examples[args.offset::args.stride]
        print(f"[run_eval] strided to {len(examples)}/{n_before} examples "
              f"(stride={args.stride}, offset={args.offset})")
    if args.limit:
        examples = examples[: args.limit]
        print(f"[run_eval] limited to first {len(examples)} examples")

    few_shots = []
    if args.n_few_shots > 0:
        if not args.few_shot_dir:
            print("error: --n-few-shots > 0 requires --few-shot-dir (eval set has no "
                  "train split to draw from).", file=sys.stderr)
            return 2
        fs_examples = load_examples(args.few_shot_dir)
        if len(fs_examples) < args.n_few_shots:
            print(f"error: --few-shot-dir has {len(fs_examples)} examples but "
                  f"--n-few-shots={args.n_few_shots}", file=sys.stderr)
            return 1
        few_shots = load_few_shots(fs_examples[: args.n_few_shots])
    print(f"[run_eval] using {len(few_shots)} few-shot examples")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run_eval] artifacts under {run_dir}")
    print(f"[run_eval] solvers: {', '.join(args.solver_models)}")
    print(f"[run_eval] validator={args.validator_model} "
          f"verifier={'off' if args.no_verifier else args.verifier_model}")

    n_outer = args.max_model_workers or max(1, len(args.solver_models))
    print(f"[run_eval] outer parallelism: {n_outer} models concurrent · "
          f"inner: {args.max_workers} examples per model")

    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(msg, flush=True)

    rows: list[tuple[str, Metrics | None, str | None, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_outer, thread_name_prefix="model"
    ) as pool:
        futures = {
            pool.submit(
                _run_one_model,
                solver_model=m,
                examples=examples,
                few_shots=few_shots,
                args=args,
                openrouter_api_key=openrouter_api_key,
                google_api_key=google_api_key,
                gen_config=gen_config,
                prior_iterations_by_id=prior_by_solver.get(m),
                run_dir=run_dir,
                log=log,
            ): m
            for m in args.solver_models
        }
        for fut in concurrent.futures.as_completed(futures):
            rows.append(fut.result())

    # Stable ordering for output
    order = {m: i for i, m in enumerate(args.solver_models)}
    rows.sort(key=lambda r: order.get(r[0], 1_000_000))

    summary_lines = [
        f"MOSAIC eval — {timestamp}",
        f"data_dir: {args.data_dir}  (n_examples={len(examples)})",
        f"validator: {args.validator_model}  "
        f"verifier: {'off' if args.no_verifier else args.verifier_model}",
        f"max_iterations={args.max_iterations}  n_few_shots={len(few_shots)}",
        "",
        f"{'model':50s} {'acc%':>8s} {'valid%':>8s} {'correct':>8s} {'valid':>8s} {'total':>6s}  report",
        "-" * 130,
    ]
    for model, m, report_path, err in rows:
        if err:
            summary_lines.append(f"{model:50s}  ERROR: {err}")
            continue
        assert m is not None
        summary_lines.append(
            f"{model:50s} {m.final_accuracy:>7.2f}% {m.final_validity:>7.2f}% "
            f"{m.final_correct:>8d} {m.final_valid:>8d} {m.total:>6d}  {report_path}"
        )

    summary_text = "\n".join(summary_lines) + "\n"
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    print()
    print(summary_text)

    # Per-model detailed summaries (verifier failure reasons etc.)
    for model, m, _path, err in rows:
        if err or m is None:
            continue
        print(f"\n=== {model} ===")
        print(format_summary(m))

    print(f"\n[run_eval] artifacts: {run_dir}")
    return 0 if all(err is None for _, _, _, err in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
