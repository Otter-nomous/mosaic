"""Command-line entry point for running a MOSAIC experiment."""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

from google import genai

from . import prompts
from .agents import Solver, Validator, Verifier
from .data import load_examples, split_examples
from .metrics import compute as compute_metrics, format_summary
from .pipeline import Pipeline, PipelineConfig, load_few_shots
from .reporting import ReportPaths, write_report, write_results_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSAIC: visual completion of chemical reactions.")
    p.add_argument("--data-dir", required=True, help="Root directory containing example subfolders.")
    p.add_argument("--output-dir", default="./runs", help="Where to write report folders.")
    p.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"),
                   help="Gemini API key (defaults to $GOOGLE_API_KEY).")

    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--n-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-few-shots", type=int, default=6,
                   help="How many train examples to show the solver as few-shot demos.")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--max-workers", type=int, default=8)

    p.add_argument("--solver-model", default="gemini-3-pro-image-preview")
    p.add_argument("--validator-model", default="gemini-3-flash-preview")
    p.add_argument("--verifier-model", default="gemini-3-flash-preview")
    p.add_argument("--no-verifier", action="store_true",
                   help="Disable the verifier (no reference comparison).")

    p.add_argument("--split", choices=["train", "val", "test"], default="val",
                   help="Which split to evaluate on.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("error: API key missing. Set $GOOGLE_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

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

    client = genai.Client(api_key=args.api_key)
    config = PipelineConfig(
        solver_prompt=prompts.SOLVER_PROMPT,
        validator_prompt=prompts.VALIDATOR_PROMPT,
        verifier_prompt=None if args.no_verifier else prompts.VERIFIER_PROMPT,
        max_iterations=args.max_iterations,
        max_workers=args.max_workers,
    )
    pipeline = Pipeline(
        solver=Solver(client, args.solver_model),
        validator=Validator(client, args.validator_model, prompts.VALIDATOR_PROMPT),
        verifier=None if args.no_verifier else Verifier(client, args.verifier_model, prompts.VERIFIER_PROMPT),
        config=config,
        few_shots=few_shots,
    )

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
