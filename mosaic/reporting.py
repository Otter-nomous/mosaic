"""Render an HTML report from a list of ExampleResult records."""

from __future__ import annotations

import base64
import html
import io
import json
import os
from dataclasses import dataclass
from typing import Optional

import PIL.Image

from .metrics import Metrics, format_summary
from .schemas import ExampleResult, IterationRecord


_MAIN_STYLE = """
body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #333; }
h1, h2, h3, h4 { color: #0056b3; }
h1 { text-align: center; }
.container { max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,.1); }
.prompt { background: #e9ecef; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; overflow-x: auto; margin-bottom: 20px; }
.example-images { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
.example-images img { max-width: 200px; height: auto; border: 1px solid #ccc; border-radius: 4px; }
.iter-block { border: 1px solid #eee; padding: 10px; margin-top: 15px; background: #fcfcfc; }
.verdict { background: #ffeeba; border: 1px solid #ffc107; color: #856404; padding: 10px; border-radius: 4px; margin-top: 10px; }
.validation { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 10px; border-radius: 4px; margin-top: 10px; }
.metrics { background: #f0f8ff; border: 1px solid #b0e0e6; padding: 15px; border-radius: 5px; margin-top: 20px; }
pre { background: #e9ecef; padding: 10px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; overflow-x: auto; }
ul { line-height: 1.7; }
"""


def _b64(image: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _img_tag(image: Optional[PIL.Image.Image], alt: str) -> str:
    if image is None:
        return f"<p><i>{alt}: not available</i></p>"
    return f'<img src="data:image/png;base64,{_b64(image)}" alt="{alt}">'


@dataclass
class ReportPaths:
    run_dir: str
    main: str

    @classmethod
    def make(cls, run_dir: str) -> "ReportPaths":
        os.makedirs(run_dir, exist_ok=True)
        return cls(run_dir=run_dir, main=os.path.join(run_dir, "main_report.html"))


def render_iteration(it: IterationRecord) -> str:
    return f"""
    <div class="iter-block">
      <h3>Iteration {it.iteration}</h3>
      <details>
        <summary>Solver prompt used</summary>
        <pre>{html.escape(it.solver_prompt_text)}</pre>
      </details>
      <div class="example-images">{_img_tag(it.solver_output_image, f"Predicted X (iter {it.iteration})")}</div>
      <div class="validation">
        <h4>Validator</h4>
        <pre>{html.escape(it.validation.model_dump_json(indent=2))}</pre>
      </div>
      <div class="verdict">
        <h4>Verifier</h4>
        <pre>{html.escape(it.verification.model_dump_json(indent=2))}</pre>
      </div>
    </div>
    """


def render_example_page(result: ExampleResult, prompts: dict[str, str]) -> str:
    images = [PIL.Image.open(p) for p in result.input_paths]
    a, b, c, d = images[0], images[1], images[2], images[3]
    iter_html = "".join(render_iteration(it) for it in result.iterations)
    error_html = (
        f'<p style="color:#a00"><b>Error:</b> {html.escape(result.error)}</p>'
        if result.error
        else ""
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Example {result.example_id}</title>
<style>{_MAIN_STYLE}</style></head>
<body><div class="container">
  <h1>Example {result.example_id}</h1>
  <p><a href="main_report.html">&#8592; back to main report</a></p>
  {error_html}
  <h2>Inputs</h2>
  <div class="example-images">
    <div><p>A</p>{_img_tag(a, "A")}</div>
    <div><p>B</p>{_img_tag(b, "B")}</div>
    <div><p>C</p>{_img_tag(c, "C")}</div>
    <div><p>Reference D</p>{_img_tag(d, "D")}</div>
  </div>
  <h2>Iterations</h2>
  {iter_html}
</div></body></html>
"""


def render_main_report(
    results: list[ExampleResult],
    metrics: Metrics,
    prompts: dict[str, str],
) -> str:
    links = "".join(
        f'<li><a href="example_{r.example_id}.html">example_{r.example_id}</a> '
        f'— {"✓" if r.is_correct else "✗"} verifier, '
        f'{"✓" if r.is_valid else "✗"} validator</li>'
        for r in results
    )
    prompt_blocks = "".join(
        f"<h2>{name} prompt</h2><div class=\"prompt\">{html.escape(text)}</div>"
        for name, text in prompts.items()
        if text
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MOSAIC Experiment Report</title>
<style>{_MAIN_STYLE}</style></head>
<body><div class="container">
  <h1>MOSAIC Experiment Report</h1>
  <div class="metrics">
    <h2>Metrics summary</h2>
    <pre>{html.escape(format_summary(metrics))}</pre>
  </div>
  <h2>Examples</h2>
  <ul>{links}</ul>
  {prompt_blocks}
</div></body></html>
"""


def write_report(
    paths: ReportPaths,
    results: list[ExampleResult],
    metrics: Metrics,
    prompts: dict[str, str],
) -> None:
    for r in results:
        with open(os.path.join(paths.run_dir, f"example_{r.example_id}.html"), "w", encoding="utf-8") as f:
            f.write(render_example_page(r, prompts))
    with open(paths.main, "w", encoding="utf-8") as f:
        f.write(render_main_report(results, metrics, prompts))


def write_results_json(paths: ReportPaths, results: list[ExampleResult]) -> None:
    """Persist a JSON view of the results (without binary images) for downstream analysis."""
    data = []
    for r in results:
        data.append({
            "example_id": r.example_id,
            "input_paths": r.input_paths,
            "error": r.error,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "validation": it.validation.model_dump(),
                    "verification": it.verification.model_dump(),
                    "solver_produced_image": it.solver_output_image is not None,
                }
                for it in r.iterations
            ],
        })
    with open(os.path.join(paths.run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
