"""End-to-end test on ONE solver model with full diagnostic output.

Catches: wrong model ID, missing modalities, unexpected response shape,
JSON-mode failures, image extraction failures.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import httpx
import PIL.Image

from mosaic import prompts
from mosaic.agents import (
    ImagePart,
    OpenRouterBackend,
    TextPart,
    _extract_openrouter_image,
)
from mosaic.data import load_examples
from mosaic.models import OPENROUTER_BASE_URL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mosaic/data")
    ap.add_argument("--model", default="google/gemini-2.5-flash-image")
    args = ap.parse_args()

    api_key = open(os.path.expanduser("~/.openrouter_key")).read().strip()
    backend = OpenRouterBackend(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    examples = load_examples(args.data_dir)
    if not examples:
        print("no examples")
        return 1
    ex = examples[0]
    print(f"[diag] example: {ex.example_id}")

    image_a = PIL.Image.open(ex.image_a)
    image_b = PIL.Image.open(ex.image_b)
    image_c = PIL.Image.open(ex.image_c)

    # Substitute placeholders in solver prompt manually with image-tagged text;
    # then send a, b, c plus the prompt as a multimodal user message.
    template = prompts.SOLVER_PROMPT
    # The prompt has $IMAGE_A/$B/$C/$FEW_SHOT/$CORRECTION placeholders. Strip
    # the few-shot/correction blocks (we're zero-shot, no history) and replace
    # image placeholders with marker text.
    text_parts = []
    contents = []
    for chunk in template.replace("$FEW_SHOT_DEMOSTRATION", "").replace("$CORRECTION_BLOCK", "").split("$IMAGE_"):
        if not chunk:
            continue
        if chunk[0] in ("A", "B", "C") and (len(chunk) == 1 or not chunk[1].isalnum()):
            tag = chunk[0]
            rest = chunk[1:]
            img = {"A": image_a, "B": image_b, "C": image_c}[tag]
            contents.append(ImagePart(img))
            if rest:
                contents.append(TextPart(rest))
        else:
            contents.append(TextPart("$IMAGE_" + chunk))

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": OpenRouterBackend._content_array(contents)}],
        "modalities": ["image", "text"],
    }
    print(f"[diag] sending {len(contents)} content parts to {args.model}")
    print(f"[diag] payload size: {len(json.dumps(payload))} bytes")

    r = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=httpx.Timeout(300, read=300),
    )
    print(f"[diag] HTTP {r.status_code}")
    if r.status_code >= 400:
        print(r.text[:2000])
        return 1

    body = r.json()
    # Print structure (without dumping image bytes)
    msg = body["choices"][0]["message"]
    print(f"[diag] message keys: {list(msg.keys())}")
    if "content" in msg:
        if isinstance(msg["content"], str):
            print(f"[diag] content (str): {msg['content'][:300]!r}")
        elif isinstance(msg["content"], list):
            print(f"[diag] content list, {len(msg['content'])} parts:")
            for p in msg["content"]:
                if isinstance(p, dict):
                    keys = list(p.keys())
                    if p.get("type") == "image_url":
                        url = (p.get("image_url") or {}).get("url", "")
                        print(f"  - image_url type, url len={len(url)}, prefix={url[:50]!r}")
                    else:
                        print(f"  - keys={keys} type={p.get('type')}")
    if "images" in msg:
        imgs = msg["images"]
        print(f"[diag] images field present, {len(imgs)} entries")
        for i, entry in enumerate(imgs):
            if isinstance(entry, dict):
                print(f"  entry[{i}] keys: {list(entry.keys())}")
                if "image_url" in entry:
                    url = entry["image_url"].get("url", "")
                    print(f"    url len={len(url)}, prefix={url[:50]!r}")
            elif isinstance(entry, str):
                print(f"  entry[{i}] str len={len(entry)}, prefix={entry[:50]!r}")

    extracted = _extract_openrouter_image(body)
    print(f"[diag] extracted image: {extracted}")
    if extracted is not None:
        print(f"[diag] image size: {extracted.size}")
        out = "/tmp/diag_solver_output.png"
        extracted.save(out)
        print(f"[diag] saved to {out}")

    return 0 if extracted else 2


if __name__ == "__main__":
    sys.exit(main())
