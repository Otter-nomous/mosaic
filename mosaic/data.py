"""Dataset loading and train/val/test splitting.

Each example is a directory containing four images named so that sorting by filename
gives the order (A, B, C, D):
    A — template reactant
    B — template product
    C — target product
    D — ground-truth target reactant (the answer)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ReactionExample:
    example_id: str
    image_a: str
    image_b: str
    image_c: str
    image_d: str

    @property
    def input_paths(self) -> list[str]:
        return [self.image_a, self.image_b, self.image_c]

    @property
    def all_paths(self) -> list[str]:
        return [self.image_a, self.image_b, self.image_c, self.image_d]


@dataclass
class Split:
    train: list[ReactionExample]
    val: list[ReactionExample]
    test: list[ReactionExample]

    def __iter__(self) -> Iterator[tuple[str, list[ReactionExample]]]:
        yield "train", self.train
        yield "val", self.val
        yield "test", self.test


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def load_examples(root: str) -> list[ReactionExample]:
    """Load every subdirectory of `root` that contains exactly 4 image files.

    Non-image files (e.g. `thoughts` text notes alongside the PNGs) are ignored.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                continue
            grouped[os.path.basename(dirpath)].append(os.path.join(dirpath, fname))

    examples = []
    skipped = 0
    for example_id in sorted(grouped):
        paths = sorted(grouped[example_id])
        if len(paths) != 4:
            skipped += 1
            continue
        examples.append(
            ReactionExample(
                example_id=example_id,
                image_a=paths[0],
                image_b=paths[1],
                image_c=paths[2],
                image_d=paths[3],
            )
        )
    if skipped:
        print(f"[data] skipped {skipped} directories without exactly 4 files")
    print(f"[data] loaded {len(examples)} examples from {root}")
    return examples


def split_examples(
    examples: list[ReactionExample],
    n_train: int,
    n_val: int = 0,
    n_test: int = 0,
    seed: int = 42,
) -> Split:
    """Deterministic random split. Caps each subset at the available count."""
    rng = random.Random(seed)
    keys = [e.example_id for e in examples]
    rng.shuffle(keys)
    by_id = {e.example_id: e for e in examples}

    total_requested = n_train + n_val + n_test
    if total_requested > len(keys):
        print(
            f"[data] requested {total_requested} examples but only {len(keys)} available; "
            "trimming proportionally"
        )

    n_train = min(n_train, len(keys))
    remaining = len(keys) - n_train
    n_val = min(n_val, remaining)
    remaining -= n_val
    n_test = min(n_test, remaining)

    train = [by_id[k] for k in keys[:n_train]]
    val = [by_id[k] for k in keys[n_train : n_train + n_val]]
    test = [by_id[k] for k in keys[n_train + n_val : n_train + n_val + n_test]]

    print(f"[data] split: train={len(train)} val={len(val)} test={len(test)}")
    return Split(train=train, val=val, test=test)
