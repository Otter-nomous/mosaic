# MOSAIC

Visual completion of chemical reaction schemes.

Given three input images:

- **A** — template reactant
- **B** — template product
- **C** — target product

…predict an image of the target reactant **X** such that `X → C` follows the same
transformation rule as `A → B`. Three Gemini agents drive the pipeline:

- **Solver** generates `X` (multimodal in, image out).
- **Validator** does a forward-simulation check on `X → C` without seeing the answer.
- **Verifier** compares `X` to the held-out reference `D` for evaluation.

Iteration is driven by validator feedback: if the validator rejects `X`, its
explanation is fed back into the solver for another attempt, up to
`--max-iterations`.

## Layout

```
mosaic/
  schemas.py     ValidationResult, VerificationResult, IterationRecord, ExampleResult
  prompts.py     SOLVER_PROMPT, VALIDATOR_PROMPT, VERIFIER_PROMPT
  data.py        ReactionExample + load_examples / split_examples
  agents.py      Solver, Validator, Verifier (thin wrappers around google-genai)
  pipeline.py    Pipeline, FewShotExample, placeholder substitution
  metrics.py     compute() + format_summary()
  reporting.py   write_report(), write_results_json()
  cli.py         argparse runner
```

## Dataset format

Point `--data-dir` at a directory whose subfolders each contain four image files.
When the four filenames are sorted lexicographically, they should be in order
`A, B, C, D`. The folder name is used as the example id.

```
datasets/
  00000001/
    a.png
    b.png
    c.png
    d.png
  00000002/
    ...
```

## Running

```sh
pip install -e .
export GOOGLE_API_KEY=...

mosaic \
  --data-dir /path/to/datasets \
  --output-dir ./runs \
  --n-train 30 --n-val 32 --n-few-shots 6 \
  --max-iterations 5 \
  --split val
```

Each run writes to `./runs/run_<timestamp>/` containing:

- `main_report.html` — metrics + index of examples
- `example_<id>.html` — per-example iteration trace with images and JSON verdicts
- `results.json` — structured results (no images) for downstream analysis

## Programmatic use

```python
from google import genai
import mosaic

examples = mosaic.load_examples("/path/to/datasets")
split = mosaic.split_examples(examples, n_train=30, n_val=32)

client = genai.Client(api_key="...")
config = mosaic.PipelineConfig(
    solver_prompt=mosaic.prompts.SOLVER_PROMPT,
    validator_prompt=mosaic.prompts.VALIDATOR_PROMPT,
    verifier_prompt=mosaic.prompts.VERIFIER_PROMPT,
    max_iterations=5,
)
pipeline = mosaic.Pipeline(
    solver=mosaic.Solver(client, "gemini-3-pro-image-preview"),
    validator=mosaic.Validator(client, "gemini-3-flash-preview", config.validator_prompt),
    verifier=mosaic.Verifier(client, "gemini-3-flash-preview", config.verifier_prompt),
    config=config,
    few_shots=mosaic.load_few_shots(split.train[:6]),
)

results = pipeline.run_dataset(split.val)
metrics = mosaic.compute(results)
print(mosaic.format_summary(metrics))
```

## Notes for porting from the Colab originals

- Colab-only deps (`google.colab.drive`, `userdata`, `!apt-get`, `/content/...`) are gone.
  If running on Colab, mount Drive yourself and pass the mounted path via `--data-dir`.
- HTML is regenerated from structured `ExampleResult` data, not parsed back as before.
  Metrics live in `metrics.py` and operate directly on `ExampleResult`s.
- `MetricAnalyzer.parse(html)` round-trip is gone; metrics are computed inline.
- The placeholder-substitution scheme has been simplified to `$IMAGE_A`, `$IMAGE_B`,
  `$IMAGE_C`, `$FEW_SHOT_DEMOSTRATION`, `$CORRECTION_BLOCK`.
- Notebook-PDF export is dropped. If you need it, run `jupyter nbconvert` manually.
