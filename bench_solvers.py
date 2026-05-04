"""Benchmark each curated OpenRouter solver model on a single example.

Single-shot: max_iterations=1, n_few_shots=0, default validator + verifier.
Per-model output: validator verdict (plausibility) and verifier verdict (vs. ground truth).

Writes a report directory under --output-dir/bench_<timestamp>/ containing:
  - results.json     : per-model verdicts + timings + explanations
  - summary.txt      : human-readable table
  - report.html      : self-contained HTML page with images + verdicts
  - <safe-model>.png : the image each solver produced (when one was returned)
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import pathlib
import sys
import time
import traceback

from mosaic import prompts
from mosaic.agents import OpenRouterBackend, Solver, Validator, Verifier
from mosaic.data import load_examples
from mosaic.models import (
    DEFAULT_VALIDATOR_OPENROUTER,
    DEFAULT_VERIFIER_OPENROUTER,
    OPENROUTER_BASE_URL,
    SOLVER_MODELS_OPENROUTER,
)
from mosaic.pipeline import Pipeline, PipelineConfig


def _safe_filename(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--validator-model", default=DEFAULT_VALIDATOR_OPENROUTER)
    ap.add_argument("--verifier-model", default=DEFAULT_VERIFIER_OPENROUTER)
    ap.add_argument("--models", nargs="*", default=SOLVER_MODELS_OPENROUTER,
                    help="Solver models to benchmark (default: all curated).")
    ap.add_argument("--gen-config", default=None,
                    help="Path to a JSON file with sampling params for OpenRouter calls. "
                         "Shape: {\"default\": {...}, \"per_model\": {model: {...}}}. "
                         "A null value in per_model opts that key out of the default.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Trials per model, executed back-to-back (model-major order) "
                         "so prompt-cache TTLs hold across repeats.")
    args = ap.parse_args()

    gen_config = None
    if args.gen_config:
        with open(os.path.expanduser(args.gen_config), "r", encoding="utf-8") as f:
            gen_config = json.load(f)

    with open(os.path.expanduser(args.api_key_file), "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    if not api_key:
        print("error: api key file is empty", file=sys.stderr)
        return 2

    backend = OpenRouterBackend(api_key=api_key, base_url=OPENROUTER_BASE_URL, gen_config=gen_config)

    examples = load_examples(args.data_dir)
    if not examples:
        print("error: no examples found in --data-dir", file=sys.stderr)
        return 1
    if len(examples) > 1:
        print(f"[bench] {len(examples)} examples available; using only the first")
    example = examples[0]
    examples = [example]

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"bench_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[bench] writing artifacts to {out_dir}")

    config = PipelineConfig(
        solver_prompt=prompts.SOLVER_PROMPT,
        validator_prompt=prompts.VALIDATOR_PROMPT,
        verifier_prompt=prompts.VERIFIER_PROMPT,
        max_iterations=1,
        max_workers=1,
    )

    rows: list[dict] = []
    for solver_model in args.models:
        print(f"\n=== {solver_model} (repeats={args.repeats}) ===", flush=True)
        pipeline = Pipeline(
            solver=Solver(backend, solver_model),
            validator=Validator(backend, args.validator_model, prompts.VALIDATOR_PROMPT),
            verifier=Verifier(backend, args.verifier_model, prompts.VERIFIER_PROMPT),
            config=config,
            few_shots=[],
        )
        for trial in range(args.repeats):
            if args.repeats > 1:
                print(f"  --- trial {trial + 1}/{args.repeats} ---", flush=True)
            t0 = time.time()
            row: dict = {
                "model": solver_model,
                "trial_index": trial,
                "valid": None,
                "correct": None,
                "validator_explanation": "",
                "verifier_explanation": "",
                "image_path": None,
                "image_size": None,
                "elapsed_s": 0.0,
                "error": None,
            }
            try:
                results = pipeline.run_dataset(examples)
                r = results[0]
                row["elapsed_s"] = time.time() - t0
                if r.error:
                    row["error"] = r.error
                    print(f"  ERROR: {r.error}")
                else:
                    f = r.final
                    row["valid"] = f.validation.is_valid_reactant
                    row["correct"] = f.verification.is_same_chemical
                    row["validator_explanation"] = f.validation.explanation
                    row["verifier_explanation"] = f.verification.explanation
                    if f.solver_output_image is not None:
                        img_name = f"{_safe_filename(solver_model)}_t{trial}.png"
                        img_path = os.path.join(out_dir, img_name)
                        f.solver_output_image.save(img_path)
                        row["image_path"] = img_name
                        row["image_size"] = list(f.solver_output_image.size)
                        print(f"  image saved: {img_name} ({f.solver_output_image.size[0]}x{f.solver_output_image.size[1]})")
                    print(f"  valid={row['valid']} correct={row['correct']} elapsed={row['elapsed_s']:.1f}s")
            except Exception as exc:
                row["elapsed_s"] = time.time() - t0
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  EXCEPTION: {row['error']}")
                traceback.print_exc()
            rows.append(row)

    # results.json
    metadata = {
        "timestamp_utc": timestamp,
        "example_id": example.example_id,
        "input_paths": example.all_paths,
        "validator_model": args.validator_model,
        "verifier_model": args.verifier_model,
        "max_iterations": 1,
        "n_few_shots": 0,
        "gen_config": gen_config,
        "repeats": args.repeats,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, indent=2)

    # summary.txt
    summary_lines = []
    header = f"{'model':50s} {'trial':>5s} {'valid':>6s} {'correct':>8s} {'time(s)':>8s}  image"
    summary_lines.append(header)
    summary_lines.append("-" * (len(header) + 30))
    for r in rows:
        v = "—" if r["valid"] is None else str(r["valid"])
        c = "—" if r["correct"] is None else str(r["correct"])
        size = "—" if not r["image_size"] else f"{r['image_size'][0]}x{r['image_size'][1]}"
        summary_lines.append(f"{r['model']:50s} {r['trial_index']:>5d} {v:>6s} {c:>8s} {r['elapsed_s']:>8.1f}  {size}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    # report.html
    _write_html_report(out_dir, metadata, rows, example)

    print("\n\n=== SUMMARY ===")
    print("\n".join(summary_lines))
    print(f"\nartifacts: {out_dir}")
    return 0


def _write_html_report(out_dir: str, metadata: dict, rows: list[dict], example) -> None:
    def img_tag(path: str | None, label: str) -> str:
        if not path:
            return f"<em>no {label}</em>"
        return f'<img src="{html.escape(path)}" alt="{html.escape(label)}" style="max-width:300px;">'

    inputs = [
        ("A (template reactant)", example.image_a),
        ("B (template product)", example.image_b),
        ("C (target product)", example.image_c),
        ("D (ground-truth reactant)", example.image_d),
    ]
    inputs_html = "".join(
        f'<div><h4>{html.escape(label)}</h4><img src="{html.escape(pathlib.Path(path).resolve().as_uri())}" style="max-width:280px;"></div>'
        for label, path in inputs
    )

    rows_html = []
    for r in rows:
        v = "—" if r["valid"] is None else ("✓" if r["valid"] else "✗")
        c = "—" if r["correct"] is None else ("✓" if r["correct"] else "✗")
        img_html = img_tag(r["image_path"], "solver output")
        size = "—" if not r["image_size"] else f"{r['image_size'][0]}×{r['image_size'][1]}"
        err = f'<div style="color:#b00;">error: {html.escape(r["error"])}</div>' if r["error"] else ""
        rows_html.append(f"""
        <tr>
          <td><code>{html.escape(r['model'])}</code></td>
          <td style="text-align:center;">{r['trial_index']}</td>
          <td style="text-align:center;font-size:1.4em;">{v}</td>
          <td style="text-align:center;font-size:1.4em;">{c}</td>
          <td>{r['elapsed_s']:.1f}s</td>
          <td>{size}</td>
          <td>{img_html}</td>
          <td><details><summary>validator</summary><pre>{html.escape(r['validator_explanation'])}</pre></details>
              <details><summary>verifier</summary><pre>{html.escape(r['verifier_explanation'])}</pre></details>
              {err}</td>
        </tr>""")

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MOSAIC bench {metadata['timestamp_utc']}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px; border-bottom: 1px solid #ccc; vertical-align: top; }}
  th {{ background: #f4f4f4; text-align: left; }}
  pre {{ white-space: pre-wrap; max-width: 600px; }}
  .meta {{ color: #555; font-size: 0.9em; }}
  .inputs {{ display: flex; gap: 1em; flex-wrap: wrap; }}
</style></head>
<body>
<h1>MOSAIC solver bench — {html.escape(metadata['timestamp_utc'])}</h1>
<p class="meta">
  example: <code>{html.escape(metadata['example_id'])}</code> ·
  validator: <code>{html.escape(metadata['validator_model'])}</code> ·
  verifier: <code>{html.escape(metadata['verifier_model'])}</code> ·
  max_iterations={metadata['max_iterations']} · n_few_shots={metadata['n_few_shots']} · repeats={metadata.get('repeats', 1)}
</p>
<h2>Inputs</h2>
<div class="inputs">{inputs_html}</div>
<h2>Per-model results</h2>
<table>
  <thead><tr>
    <th>model</th><th>trial</th><th>valid</th><th>correct</th><th>time</th><th>output size</th><th>output image</th><th>explanations</th>
  </tr></thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html_doc)


if __name__ == "__main__":
    sys.exit(main())
