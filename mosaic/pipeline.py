"""Iterative refinement pipeline: solver → validator → verifier, with feedback loop."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import PIL.Image
from google.genai import types
from tqdm import tqdm

from .agents import Solver, Validator, Verifier, pil_to_part
from .data import ReactionExample
from .schemas import ExampleResult, IterationRecord, ValidationResult, VerificationResult


@dataclass
class FewShotExample:
    """One worked example shown to the solver before the actual problem."""

    image_a: PIL.Image.Image
    image_b: PIL.Image.Image
    image_c: PIL.Image.Image
    image_d: PIL.Image.Image


def load_few_shots(examples: list[ReactionExample]) -> list[FewShotExample]:
    return [
        FewShotExample(
            image_a=PIL.Image.open(e.image_a),
            image_b=PIL.Image.open(e.image_b),
            image_c=PIL.Image.open(e.image_c),
            image_d=PIL.Image.open(e.image_d),
        )
        for e in examples
    ]


def _build_few_shot_parts(few_shots: list[FewShotExample]) -> list[types.Part]:
    parts: list[types.Part] = []
    if not few_shots:
        return parts
    parts.append(types.Part(text="\n### FEW-SHOT EXAMPLES ###\n"))
    for i, fs in enumerate(few_shots, 1):
        parts.append(types.Part(text=f"\nExample {i}:\nReaction Template (A):"))
        parts.append(pil_to_part(fs.image_a))
        parts.append(types.Part(text="Product Template (B):"))
        parts.append(pil_to_part(fs.image_b))
        parts.append(types.Part(text="Target Product (C):"))
        parts.append(pil_to_part(fs.image_c))
        parts.append(types.Part(text="Expected Reactant X (D):"))
        parts.append(pil_to_part(fs.image_d))
        parts.append(types.Part(text="\n---\n"))
    parts.append(types.Part(text="### END FEW-SHOT EXAMPLES ###\n\n"))
    return parts


def _build_correction_block(history: list[IterationRecord]) -> tuple[list[types.Part], str]:
    """Compose feedback from previous attempts into model parts plus a text trace.

    Returns (parts_for_model, text_for_report).
    """
    if not history:
        return [], ""

    parts: list[types.Part] = []
    text_lines: list[str] = []

    header = (
        f"### CORRECTION TASK (Iteration {len(history) + 1}):\n"
        "You have received feedback on your previous attempts. Review the history below "
        "and generate a corrected Reactant X that addresses the issues.\n\n"
    )
    parts.append(types.Part(text=header))
    text_lines.append(header)

    for prior in history:
        feedback = prior.validation.explanation
        line = f"**Iteration {prior.iteration} feedback:** \"{feedback}\"\n"
        parts.append(types.Part(text=line))
        text_lines.append(line)

        if prior.solver_output_image is not None:
            parts.append(types.Part(text=f"Image generated in iteration {prior.iteration}:\n"))
            parts.append(pil_to_part(prior.solver_output_image))
            text_lines.append(f"Image generated in iteration {prior.iteration}:\n<IMAGE>\n")
        else:
            line = f"No image was generated in iteration {prior.iteration}.\n"
            parts.append(types.Part(text=line))
            text_lines.append(line)

        parts.append(types.Part(text="---\n"))
        text_lines.append("---\n")

    closing = (
        "Re-evaluate the original inputs and generate a corrected Reactant X that resolves "
        "all feedback while still adhering to the drawing standards.\n"
    )
    parts.append(types.Part(text=closing))
    text_lines.append(closing)

    return parts, "".join(text_lines)


_PLACEHOLDER_PATTERN = re.compile(
    r"(\$IMAGE_A|\$IMAGE_B|\$IMAGE_C|\$FEW_SHOT_DEMOSTRATION|\$CORRECTION_BLOCK)"
)


def _build_solver_contents(
    template: str,
    image_a: PIL.Image.Image,
    image_b: PIL.Image.Image,
    image_c: PIL.Image.Image,
    few_shot_parts: list[types.Part],
    correction_parts: list[types.Part],
    correction_text: str,
) -> tuple[list[types.Part], str]:
    """Substitute placeholders in the template with real content + a textual trace."""
    contents: list[types.Part] = []
    text_trace: list[str] = []
    image_map = {"$IMAGE_A": image_a, "$IMAGE_B": image_b, "$IMAGE_C": image_c}

    for segment in _PLACEHOLDER_PATTERN.split(template):
        if segment in image_map:
            contents.append(pil_to_part(image_map[segment]))
            text_trace.append(f"<{segment[1:]}>")
        elif segment == "$FEW_SHOT_DEMOSTRATION":
            contents.extend(few_shot_parts)
            text_trace.append("<FEW_SHOT_BLOCK>" if few_shot_parts else "")
        elif segment == "$CORRECTION_BLOCK":
            contents.extend(correction_parts)
            text_trace.append(correction_text)
        elif segment:
            contents.append(types.Part(text=segment))
            text_trace.append(segment)

    return contents, "".join(text_trace)


@dataclass
class PipelineConfig:
    solver_prompt: str
    validator_prompt: str
    verifier_prompt: Optional[str] = None
    max_iterations: int = 5
    max_workers: int = 8


class Pipeline:
    """Runs the full solver → validator → verifier loop over a dataset."""

    def __init__(
        self,
        solver: Solver,
        validator: Validator,
        verifier: Optional[Verifier],
        config: PipelineConfig,
        few_shots: Optional[list[FewShotExample]] = None,
    ):
        self.solver = solver
        self.validator = validator
        self.verifier = verifier
        self.config = config
        self.few_shots = few_shots or []
        self._few_shot_parts = _build_few_shot_parts(self.few_shots)

    def run_example(self, example: ReactionExample) -> ExampleResult:
        try:
            image_a = PIL.Image.open(example.image_a)
            image_b = PIL.Image.open(example.image_b)
            image_c = PIL.Image.open(example.image_c)
            reference_d = PIL.Image.open(example.image_d)
        except Exception as exc:
            return ExampleResult(
                example_id=example.example_id,
                input_paths=example.all_paths,
                error=f"image load failed: {exc}",
            )

        result = ExampleResult(example_id=example.example_id, input_paths=example.all_paths)

        for i in range(1, self.config.max_iterations + 1):
            correction_parts, correction_text = _build_correction_block(result.iterations)
            contents, prompt_text = _build_solver_contents(
                self.config.solver_prompt,
                image_a, image_b, image_c,
                self._few_shot_parts,
                correction_parts,
                correction_text,
            )

            output = self.solver.run(contents)

            if output is None:
                validation = ValidationResult(
                    is_valid_reactant=False,
                    explanation=f"Iteration {i}: solver failed to generate an image.",
                )
                verification = VerificationResult(
                    is_same_chemical=False,
                    explanation="Skipped because solver produced no image.",
                )
            else:
                validation = self._safe_validate(image_a, image_b, image_c, output, i)
                verification = self._verify_or_default(output, reference_d, validation)

            result.iterations.append(IterationRecord(
                iteration=i,
                solver_output_image=output,
                solver_prompt_text=prompt_text,
                validation=validation,
                verification=verification,
            ))

            if validation.is_valid_reactant:
                break

        return result

    def _safe_validate(self, a, b, c, x, i) -> ValidationResult:
        try:
            return self.validator.run(a, b, c, x)
        except Exception as exc:
            return ValidationResult(
                is_valid_reactant=False,
                explanation=f"Iteration {i}: validator error: {exc}",
            )

    def _verify_or_default(self, predicted, reference, validation) -> VerificationResult:
        if not validation.is_valid_reactant:
            return VerificationResult(
                is_same_chemical=False,
                explanation=f"Validation failed: {validation.explanation}",
            )
        if self.verifier is None:
            return VerificationResult(
                is_same_chemical=True,
                explanation="Verifier disabled; passing because validation succeeded.",
            )
        try:
            return self.verifier.run(predicted, reference)
        except Exception as exc:
            return VerificationResult(
                is_same_chemical=False,
                explanation=f"Verifier error: {exc}",
            )

    def run_dataset(self, examples: list[ReactionExample]) -> list[ExampleResult]:
        results: list[ExampleResult] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self.run_example, ex): ex for ex in examples}
            for future in tqdm(as_completed(futures), total=len(futures), desc="examples"):
                ex = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(ExampleResult(
                        example_id=ex.example_id,
                        input_paths=ex.all_paths,
                        error=f"pipeline crashed: {exc}",
                    ))
        results.sort(key=lambda r: r.example_id)
        return results
