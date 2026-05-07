"""Extract per-iteration solver_output_image PNGs from existing example_*.html
reports into <run-dir>/<solver>/iter_images/<example_id>/iter_<N>.png.

One-shot helper for resuming runs whose original execution didn't persist images.
Going forward, run_eval.py writes these PNGs directly so this isn't needed.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from glob import glob


_ITER_BLOCK = re.compile(
    r'<h3>Iteration\s+(\d+)</h3>(.*?)(?=<h3>Iteration\s+\d+</h3>|</div></body>)',
    re.DOTALL,
)
# In each iteration block, the FIRST <img src="data:image/png;base64,...">
# is the solver output (input images A/B/C/D live above the iterations block).
_FIRST_IMG = re.compile(
    r'<div class="example-images">\s*<img src="data:image/png;base64,([^"]+)"',
)


def extract_one_html(html_path: str) -> dict[int, bytes | None]:
    """Return {iter_number: png_bytes_or_None_if_missing} for one example HTML."""
    with open(html_path, "r", encoding="utf-8") as f:
        doc = f.read()
    out: dict[int, bytes | None] = {}
    for m in _ITER_BLOCK.finditer(doc):
        iter_n = int(m.group(1))
        block = m.group(2)
        img_match = _FIRST_IMG.search(block)
        if img_match:
            out[iter_n] = base64.b64decode(img_match.group(1))
        else:
            # solver_output_image was None — _img_tag emitted "<p><i>...not available</i></p>"
            out[iter_n] = None
    return out


def example_id_from_path(html_path: str) -> str:
    base = os.path.basename(html_path)
    # example_00000020.html -> 00000020
    return base.removeprefix("example_").removesuffix(".html")


def extract_solver_dir(solver_dir: str) -> tuple[int, int, int]:
    """Walk every example_*.html in solver_dir, write iter_images/<id>/iter_N.png.
    Returns (n_examples, n_iters_with_image, n_iters_without_image).
    """
    out_root = os.path.join(solver_dir, "iter_images")
    os.makedirs(out_root, exist_ok=True)
    n_ex = n_with = n_without = 0
    for html_path in sorted(glob(os.path.join(solver_dir, "example_*.html"))):
        ex_id = example_id_from_path(html_path)
        iters = extract_one_html(html_path)
        if not iters:
            continue
        n_ex += 1
        ex_out = os.path.join(out_root, ex_id)
        os.makedirs(ex_out, exist_ok=True)
        for iter_n, png in iters.items():
            if png is None:
                n_without += 1
                continue
            with open(os.path.join(ex_out, f"iter_{iter_n}.png"), "wb") as f:
                f.write(png)
            n_with += 1
    return n_ex, n_with, n_without


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Path to runs/run_<timestamp>/")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 2

    solver_dirs = [
        d for d in sorted(glob(os.path.join(args.run_dir, "*")))
        if os.path.isdir(d) and glob(os.path.join(d, "example_*.html"))
    ]
    if not solver_dirs:
        print(f"error: no per-solver subdirs with example_*.html under {args.run_dir}",
              file=sys.stderr)
        return 1

    grand = [0, 0, 0]
    for sd in solver_dirs:
        n_ex, n_with, n_without = extract_solver_dir(sd)
        print(f"{os.path.basename(sd)}: {n_ex} examples, "
              f"{n_with} iters with image, {n_without} without")
        grand[0] += n_ex
        grand[1] += n_with
        grand[2] += n_without

    print(f"\nTOTAL: {grand[0]} examples, {grand[1]} iter-images extracted, "
          f"{grand[2]} iters had no image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
