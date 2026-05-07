"""Diagnostic: verify OpenRouter model IDs and inspect raw response shape."""
from __future__ import annotations

import json
import os
import sys

import httpx

from mosaic.models import (
    OPENROUTER_BASE_URL,
    SOLVER_MODELS_OPENROUTER,
    VALIDATOR_MODELS_OPENROUTER,
)


def main() -> int:
    key_path = os.path.expanduser("~/.openrouter_key")
    api_key = open(key_path).read().strip()
    headers = {"Authorization": f"Bearer {api_key}"}

    # 1. List available models
    r = httpx.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=30)
    r.raise_for_status()
    available = {m["id"] for m in r.json()["data"]}
    print(f"[diag] OpenRouter exposes {len(available)} models")

    # 2. Check our curated lists
    print("\n[diag] Solver model availability:")
    for m in SOLVER_MODELS_OPENROUTER:
        mark = "OK " if m in available else "MISSING"
        print(f"  {mark}  {m}")

    print("\n[diag] Validator/Verifier model availability:")
    for m in VALIDATOR_MODELS_OPENROUTER:
        mark = "OK " if m in available else "MISSING"
        print(f"  {mark}  {m}")

    # 3. For each missing solver model, suggest near matches
    print("\n[diag] Near-matches for missing solver models:")
    for m in SOLVER_MODELS_OPENROUTER:
        if m in available:
            continue
        # crude substring search on the trailing slug
        slug = m.split("/")[-1]
        candidates = [a for a in available if slug.split("-")[0] in a and ("image" in a or "vision" in a or slug.split("-")[1] in a if "-" in slug else True)]
        candidates = sorted(candidates)[:8]
        print(f"  for {m!r}:")
        for c in candidates:
            print(f"    - {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
