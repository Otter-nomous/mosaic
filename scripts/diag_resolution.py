"""Probe each solver model to see if it honors an output-resolution hint.

For each model, fire two minimal generation requests:
  1. baseline (no size hint)
  2. with `size: "512x512"` in the request body

Records actual output dimensions both ways so we can tell whether the
hint is honored.
"""
from __future__ import annotations

import io
import os
import sys
import time

import httpx
import PIL.Image

from mosaic.agents import _extract_openrouter_image
from mosaic.models import OPENROUTER_BASE_URL, SOLVER_MODELS_OPENROUTER


PROMPT = (
    "Generate a simple line drawing of a single benzene ring (a hexagon "
    "with alternating double bonds), white background, black lines."
)


def call(api_key: str, model: str, extra: dict | None) -> tuple[int, tuple[int, int] | None, float, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
    }
    if extra:
        payload.update(extra)
    t0 = time.time()
    try:
        r = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(180, read=180),
        )
        elapsed = time.time() - t0
        if r.status_code >= 400:
            return r.status_code, None, elapsed, r.text[:200]
        body = r.json()
        img = _extract_openrouter_image(body)
        return r.status_code, (img.size if img else None), elapsed, ""
    except Exception as exc:
        return -1, None, time.time() - t0, str(exc)


def main() -> int:
    api_key = open(os.path.expanduser("~/.openrouter_key")).read().strip()
    print(f"{'model':45s} {'baseline':>15s} {'size_hint':>15s} {'b_t(s)':>7s} {'h_t(s)':>7s}  notes")
    print("-" * 110)
    for model in SOLVER_MODELS_OPENROUTER:
        s1, sz1, t1, err1 = call(api_key, model, None)
        s2, sz2, t2, err2 = call(api_key, model, {"size": "512x512"})
        b = f"{sz1[0]}x{sz1[1]}" if sz1 else f"err{s1}"
        h = f"{sz2[0]}x{sz2[1]}" if sz2 else f"err{s2}"
        notes = []
        if sz1 and sz2 and sz1 == sz2:
            notes.append("size hint NOT honored" if sz2 != (512, 512) else "honored")
        if err1: notes.append(f"baseline: {err1[:60]}")
        if err2: notes.append(f"hinted: {err2[:60]}")
        print(f"{model:45s} {b:>15s} {h:>15s} {t1:>7.1f} {t2:>7.1f}  {'; '.join(notes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
