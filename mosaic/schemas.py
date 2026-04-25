"""Structured result types for the MOSAIC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Validator verdict: does X plausibly transform into C under the rule A→B?"""

    is_valid_reactant: bool
    explanation: str
    confidence_score: Optional[int] = None


class VerificationResult(BaseModel):
    """Verifier verdict: is the predicted X topologically equivalent to the reference D?"""

    is_same_chemical: bool
    explanation: str


@dataclass
class IterationRecord:
    """One pass through solver → validator → verifier."""

    iteration: int
    solver_output_image: object  # PIL.Image.Image | None
    solver_prompt_text: str
    validation: ValidationResult
    verification: VerificationResult


@dataclass
class ExampleResult:
    """All iterations for a single dataset example."""

    example_id: str
    input_paths: list[str]
    iterations: list[IterationRecord] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def final(self) -> Optional[IterationRecord]:
        return self.iterations[-1] if self.iterations else None

    @property
    def is_correct(self) -> bool:
        f = self.final
        return bool(f and f.verification.is_same_chemical)

    @property
    def is_valid(self) -> bool:
        f = self.final
        return bool(f and f.validation.is_valid_reactant)
