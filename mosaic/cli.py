"""Command-line entry point for running a MOSAIC experiment."""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

from . import prompts
from .agents import GeminiBackend, OpenRouterBackend, Solver, Validator, Verifier
from .data import load_examples, split_examples
from .metrics import compute as compute_metrics, format_summary
from .models import (
    DEFAULT_SOLVER_GEMINI,
    DEFAULT_SOLVER_OPENROUTER,
    DEFAULT_VALIDATOR_GEMINI,
    DEFAULT_VALIDATOR_OPENROUTER,
    DEFAULT_VERIFIER_GEMINI,
    DEFAULT_VERIFIER_OPENROUTER,
    OPENROUTER_BASE_URL,
    SOLVER_MODELS_OPENROUTER,
    VALIDATOR_MODELS_OPENROUTER,
)
from .pipeline import Pipeline, PipelineConfig, load_few_shots
from .reporting import ReportPaths, write_report, write_results_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSAIC: visual completion of chemical reactions.")
    p.add_argument("--data-dir", required=True, help="Root directory containing example subfolders.")
    p.add_argument("--output-dir", default="./runs", help="Where to write report folders.")

    p.add_argument("--provider", choices=["gemini", "openrouter"], default="openrouter",
                   help="Which API backend to use for all agents.")
    p.add_argument("--api-key", default=None,
                   help="Override the provider API key (raw value). Prefer --api-key-file. "
                        "Falls back to $GOOGLE_API_KEY (gemini) or $OPENROUTER_API_KEY (openrouter).")
    p.add_argument("--api-key-file", default=None,
                   help="Path to a file whose contents are the API key. The file is read "
                        "directly by this process; the key is never logged or echoed.")
    p.add_argument("--openrouter-base-url", default=OPENROUTER_BASE_URL,
                   help="OpenRouter base URL (only used when --provider=openrouter).")

    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--n-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-few-shots", type=int, default=6,
                   help="How many train examples to show the solver as few-shot demos.")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--max-workers", type=int, default=8)

    p.add_argument("--solver-model", default=None,
                   help=f"Solver model. Default: {DEFAULT_SOLVER_OPENROUTER} (openrouter) "
                        f"or {DEFAULT_SOLVER_GEMINI} (gemini). "
                        f"Curated OpenRouter options: {', '.join(SOLVER_MODELS_OPENROUTER)}.")
    p.add_argument("--validator-model", default=None,
                   help=f"Validator model. Default: {DEFAULT_VALIDATOR_OPENROUTER} (openrouter) "
                        f"or {DEFAULT_VALIDATOR_GEMINI} (gemini). "
                        f"Curated OpenRouter options: {', '.join(VALIDATOR_MODELS_OPENROUTER)}.")
    p.add_argument("--verifier-model", default=None,
                   help=f"Verifier model. Default: {DEFAULT_VERIFIER_OPENROUTER} (openrouter) "
                        f"or {DEFAULT_VERIFIER_GEMINI} (gemini). "
                        f"Same curated list as validator.")
    p.add_argument("--no-verifier", action="store_true",
                   help="Disable the verifier (no reference comparison).")

    p.add_argument("--split", choices=["train", "val", "test"], default="val",
                   help="Which split to evaluate on.")
    return p.parse_args(argv)


def _resolve_api_key(provider: str, override: str | None, key_file: str | None) -> str | None:
    if override:
        return override
    if key_file:
        path = os.path.expanduser(key_file)
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "GOOGLE_API_KEY"
    return os.environ.get(env_var)


def _resolve_models(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.provider == "openrouter":
        solver = args.solver_model or DEFAULT_SOLVER_OPENROUTER
        validator = args.validator_model or DEFAULT_VALIDATOR_OPENROUTER
        verifier = args.verifier_model or DEFAULT_VERIFIER_OPENROUTER
    else:
        solver = args.solver_model or DEFAULT_SOLVER_GEMINI
        validator = args.validator_model or DEFAULT_VALIDATOR_GEMINI
        verifier = args.verifier_model or DEFAULT_VERIFIER_GEMINI
    return solver, validator, verifier


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = _resolve_api_key(args.provider, args.api_key, args.api_key_file)
    if not api_key:
        env_var = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "GOOGLE_API_KEY"
        print(f"error: API key missing. Set ${env_var}, pass --api-key-file, or --api-key.",
              file=sys.stderr)
        return 2

    solver_model, validator_model, verifier_model = _resolve_models(args)

    if args.provider == "openrouter":
        backend = OpenRouterBackend(api_key=api_key, base_url=args.openrouter_base_url)
    else:
        backend = GeminiBackend(api_key=api_key)

    examples = load_examples(args.data_dir)
    split = split_examples(
        examples,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    few_shot_pool = list(split.train)
    rng.shuffle(few_shot_pool)
    few_shot_examples = few_shot_pool[: args.n_few_shots]
    few_shots = load_few_shots(few_shot_examples)
    print(f"[cli] using {len(few_shots)} few-shot examples")

    eval_examples = {"train": split.train, "val": split.val, "test": split.test}[args.split]
    if not eval_examples:
        print(f"error: --split {args.split} is empty", file=sys.stderr)
        return 1

    config = PipelineConfig(
        solver_prompt=prompts.SOLVER_PROMPT,
        validator_prompt=prompts.VALIDATOR_PROMPT,
        verifier_prompt=None if args.no_verifier else prompts.VERIFIER_PROMPT,
        max_iterations=args.max_iterations,
        max_workers=args.max_workers,
    )
    pipeline = Pipeline(
        solver=Solver(backend, solver_model),
        validator=Validator(backend, validator_model, prompts.VALIDATOR_PROMPT),
        verifier=None if args.no_verifier else Verifier(backend, verifier_model, prompts.VERIFIER_PROMPT),
        config=config,
        few_shots=few_shots,
    )

    print(f"[cli] provider={args.provider} solver={solver_model} "
          f"validator={validator_model} verifier={verifier_model if not args.no_verifier else 'off'}")
    print(f"[cli] running pipeline on {len(eval_examples)} {args.split} examples")
    results = pipeline.run_dataset(eval_examples)
    metrics = compute_metrics(results)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths = ReportPaths.make(os.path.join(args.output_dir, f"run_{timestamp}"))
    prompt_map = {
        "Solver": config.solver_prompt,
        "Validator": config.validator_prompt,
        "Verifier": config.verifier_prompt or "",
    }
    write_report(paths, results, metrics, prompt_map)
    write_results_json(paths, results)

    print(format_summary(metrics))
    print(f"[cli] report written to {paths.main}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
