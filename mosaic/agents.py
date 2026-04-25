"""Three Gemini agents that drive the MOSAIC pipeline.

Solver    — generates an image of reactant X (multimodal in, image out)
Validator — checks if X→C is plausible under rule A→B (no reference needed)
Verifier  — compares predicted X to reference D (used for evaluation)
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Optional

import PIL.Image
from google import genai
from google.genai import types

from .schemas import ValidationResult, VerificationResult


def pil_to_part(image: PIL.Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=buf.getvalue()))


def _extract_image(response: Any) -> Optional[PIL.Image.Image]:
    parts = []
    if getattr(response, "parts", None):
        parts = response.parts
    elif response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts or []

    for part in parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return PIL.Image.open(io.BytesIO(part.inline_data.data))
    return None


class Solver:
    """Generates an image of reactant X."""

    def __init__(
        self,
        client: genai.Client,
        model: str,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def run(self, contents: list[types.Part]) -> Optional[PIL.Image.Image]:
        config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000)
        )
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return _extract_image(response)
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                print(f"[solver] attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        return None


class Validator:
    """Checks whether predicted X plausibly transforms to C under rule A→B."""

    def __init__(self, client: genai.Client, model: str, prompt: str):
        self.client = client
        self.model = model
        self.prompt = prompt

    def run(
        self,
        image_a: PIL.Image.Image,
        image_b: PIL.Image.Image,
        image_c: PIL.Image.Image,
        image_x: PIL.Image.Image,
    ) -> ValidationResult:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.prompt, image_a, image_b, image_c, image_x],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ValidationResult,
            ),
        )
        return ValidationResult.model_validate_json(response.text)


class Verifier:
    """Compares predicted X with reference D for evaluation."""

    def __init__(self, client: genai.Client, model: str, prompt: str):
        self.client = client
        self.model = model
        self.prompt = prompt

    def run(
        self,
        predicted: PIL.Image.Image,
        reference: PIL.Image.Image,
    ) -> VerificationResult:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.prompt, predicted, reference],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VerificationResult,
            ),
        )
        return VerificationResult.model_validate_json(response.text)
