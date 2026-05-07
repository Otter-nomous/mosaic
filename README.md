# MOSAIC

**M**ultimodal **O**ptimization with **S**olver-validator-verifier **A**gents for
**I**mage **C**ompletion of chemical reaction schemes.

Given three input images:

- **A** — template reactant
- **B** — template product
- **C** — target product

…predict an image of the target reactant **X** such that `X → C` follows the same
transformation rule as `A → B`. Three agents drive the loop:

- **Solver** (image-out, e.g. `gemini-3-pro-image-preview`) generates `X`.
- **Validator** (text-out) does a forward-simulation check on `X → C` *without*
  seeing the held-out answer. Its critique is fed back to the solver.
- **Verifier** (text-out) compares `X` against the held-out reference `D` for
  scoring. Only used to compute eval metrics — the loop never sees it.

Iteration stops when the validator says `is_valid_reactant=true` or after
`--max-iterations` rounds.

## Install

```sh
git clone https://github.com/Otter-nomous/mosaic.git
cd mosaic
./install.sh
source .venv/bin/activate
```

`install.sh` picks the highest available Python 3.10+ on `PATH`, creates
`./.venv`, and installs the package in editable mode. Re-running is safe.
If you don't have Python 3.10+: `brew install python@3.12` (macOS) or use
your distro's `python3.12` package / `pyenv` (Linux).

## API keys

Solver/validator/verifier models can be served by either provider; routing is
chosen per-model from the model name:

| Model name                          | Provider           | Key needed         |
|-------------------------------------|--------------------|--------------------|
| `gemini-...`, `gemma-...` (bare)    | Google AI Studio   | `GOOGLE_API_KEY`   |
| `google/gemini-...` (slash-prefix)  | OpenRouter         | `OPENROUTER_API_KEY` |
| Any other (`openai/...`, `anthropic/...`, `x-ai/...`, …) | OpenRouter | `OPENROUTER_API_KEY` |

Three resolution paths, in precedence order:

1. CLI flag: `--api-key-file <path>` (OpenRouter), `--google-api-key-file <path>`.
2. Environment variable: `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`.
3. Default file: `.secrets/openrouter_api_key`, `.secrets/google_api_key`
   (`.secrets/` is gitignored — paste keys there once and forget about them).

You only need keys for providers your run actually touches.

## Dataset format

Each example lives in its own directory containing four images named so that
sorting by filename gives the order `A, B, C, D`. Subdirectories of any depth
are walked. Non-image files (e.g. a `thoughts` note next to the images) are
ignored.

```
release_data/eval/
  easy/
    00000011/
      00000011_A.png   # template reactant
      00000011_B.png   # template product
      00000011_C.png   # target product
      00000011_D.png   # ground-truth target reactant (held out from solver/validator)
  medium/
    00000018/...
  hard/
    00000072/...
```

The folder name is the example id and is used throughout reports.

## Running an evaluation

The main driver is `run_eval.py` — it sweeps one or more solver models over a
dataset, with two levels of parallelism (per-model and per-example).

```sh
# Smallest useful run: 2 examples, default solvers (one OpenAI + one Gemini),
# default validator/verifier (Gemini Flash), 5 iterations.
python run_eval.py --limit 2

# Real eval: three Gemini solvers, full eval set, 8 iterations,
# 12 examples in flight per model, 1 model at a time.
python run_eval.py \
  --solver-models \
      gemini-3-pro-image-preview \
      gemini-3.1-flash-image-preview \
      gemini-2.5-flash-image \
  --validator-model gemini-3-flash-preview \
  --verifier-model  gemini-3-flash-preview \
  --max-iterations 8 \
  --max-workers 12 \
  --max-model-workers 1
```

Each run lands in `runs/run_<timestamp>/<safe-model-name>/` and contains:

- `main_report.html` — metrics + index of examples
- `example_<id>.html` — per-example iteration trace with images and JSON verdicts
- `results.json` — final structured results (no images)
- `results.jsonl` — streamed checkpoint, one example per line, written as the
  run progresses (so a kill -9 leaves something useful behind)
- `iter_images/<example_id>/iter_<N>.png` — every solver image produced, used
  to make resumes possible

A top-level `summary.txt` aggregates per-model metrics across the sweep.

### Sampling parameters

`--gen-config <path>` (default: `gen_config_v3.json`) sets temperature / top_p /
seed. The schema is `{"default": {...}, "per_model": {model: {...}}}`; a `null`
entry under `per_model` opts that model out of the corresponding default
(needed for OpenAI image models that 400 on `temperature`).

```json
{
  "default":   { "temperature": 0, "seed": 42, "top_p": 1.0 },
  "per_model": { "openai/gpt-5.4-image-2": { "temperature": null, "top_p": null } }
}
```

### Resuming / extending iterations

`--resume-from <prior-run-dir>` continues an existing run instead of starting
fresh:

```sh
# A run finished at iter 5 — push it to iter 8 without re-doing iter 1-5.
python run_eval.py \
  --resume-from runs/run_20260504_174107 \
  --max-iterations 8 \
  --solver-models gemini-3-pro-image-preview
```

Per-example behavior:

- Last prior iteration was already valid → short-circuit (no API calls).
- Last prior iteration was invalid → loop resumes at `len(prior) + 1` with the
  full prior history fed back to the solver as feedback context.

The progress bar pre-sorts short-circuit examples to the front, so the bar
moves quickly through the cached tail and only the truly-pending examples gate
wall-clock.

## Helper scripts

- **`bench_solvers.py`** — single-shot sanity check across the curated solver
  list on one example. Quick way to see "does this model produce a sensible
  image at all?" before spending money on the full sweep.
- **`run_few_shots_eval.py`** — few-shot ablation. Takes a stratified held-out
  pool of K examples and sweeps shot counts ∈ {1, 3, 5, 7} on the remainder,
  one `run_eval.py` subprocess per shot count.
- **`run_validators.py`** — pin one solver image per example, then run N
  validator candidates against it (no iteration, no feedback loop). Use the
  decision matrix to pick a validator that best agrees with the verifier.
- **`compare_verifiers.py`** — re-runs a new verifier over a finished run's
  saved `iter_<n>.png` images and surfaces disagreements with the original
  verifier verdict. No solver calls.
- **`extract_iter_images.py`** — one-shot helper for runs predating per-iter
  image persistence: parses `example_*.html` and writes
  `iter_images/<id>/iter_N.png` so they can be `--resume-from`'d.

## Plots

`plots/` holds NeurIPS-style figure scripts (serif font, embeddable PDF
fonts):

- `plot_model_accuracy.py` — bar chart of final accuracy across baselines /
  vanilla solvers / our multi-agent system, with growth-arrow annotations.
- `plot_per_iteration_metrics.py` — per-iteration accuracy + validity curves
  across solvers.

Each script writes both a `.pdf` (vector, paper-ready) and a `.png` (preview).

## Programmatic use

```python
import mosaic
from mosaic.agents import make_backend

examples = mosaic.load_examples("release_data/eval")
split = mosaic.split_examples(examples, n_train=30, n_val=32)

solver_model    = "gemini-3-pro-image-preview"
validator_model = "gemini-3-flash-preview"

# make_backend auto-routes by model name + key availability.
solver_backend = make_backend(
    solver_model,
    openrouter_api_key=None,
    google_api_key="...",
    openrouter_base_url="https://openrouter.ai/api/v1",
)

config = mosaic.PipelineConfig(
    solver_prompt=mosaic.prompts.SOLVER_PROMPT,
    validator_prompt=mosaic.prompts.VALIDATOR_PROMPT,
    verifier_prompt=mosaic.prompts.VERIFIER_PROMPT,
    max_iterations=5,
)
pipeline = mosaic.Pipeline(
    solver=mosaic.Solver(solver_backend, solver_model),
    validator=mosaic.Validator(solver_backend, validator_model, config.validator_prompt),
    verifier=mosaic.Verifier(solver_backend, validator_model, config.verifier_prompt),
    config=config,
    few_shots=mosaic.load_few_shots(split.train[:6]),
)

results = pipeline.run_dataset(split.val)
print(mosaic.format_summary(mosaic.compute(results)))
```

## Layout

```
mosaic/
  schemas.py     ValidationResult, VerificationResult, IterationRecord, ExampleResult
  prompts.py     SOLVER_PROMPT, VALIDATOR_PROMPT, VERIFIER_PROMPT
  data.py        ReactionExample + load_examples / split_examples
  agents.py      Solver / Validator / Verifier + GeminiBackend / OpenRouterBackend
                 + make_backend / pick_provider / is_gemini_direct
  models.py      Curated OpenRouter + Gemini-direct model name lists
  pipeline.py    Pipeline, FewShotExample, placeholder substitution, resume
  metrics.py     compute() + format_summary()
  reporting.py   write_report(), write_results_json()
  cli.py         argparse runner for the simple one-model train/val workflow

run_eval.py            Multi-model eval driver (the one most users want)
bench_solvers.py       Single-shot solver sanity check
run_validators.py      Validator-vs-verifier agreement matrix
compare_verifiers.py   Re-score finished runs with a new verifier
run_few_shots_eval.py  Few-shot count ablation sweep
extract_iter_images.py Backfill iter_images/ for legacy runs

plots/                 NeurIPS-style figure scripts (.pdf + .png outputs)
gen_config_v3.json     Default sampling params (temperature/top_p/seed)
.secrets/              Gitignored; drop API key files here
```
