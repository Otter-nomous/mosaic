"""Solver/Validator/Verifier agents with pluggable backends.

Solver    — generates an image of reactant X (multimodal in, image out)
Validator — checks if X→C is plausible under rule A→B (no reference needed)
Verifier  — compares predicted X to reference D (used for evaluation)

Two backends are supported:
- GeminiBackend     — uses google-genai directly against the Gemini API
- OpenRouterBackend — uses httpx directly against OpenRouter's REST API
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Type, Union

import httpx
import PIL.Image
from pydantic import BaseModel

from .schemas import ValidationResult, VerificationResult


# ---------------------------------------------------------------------------
# Neutral content representation (provider-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    image: PIL.Image.Image


ContentPart = Union[TextPart, ImagePart]


def _pil_to_png_bytes(image: PIL.Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_data_url(image: PIL.Image.Image) -> str:
    b64 = base64.b64encode(_pil_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    def generate_image(
        self, model: str, contents: list[ContentPart], timeout_seconds: int
    ) -> Optional[PIL.Image.Image]: ...

    def generate_json(
        self,
        model: str,
        contents: list[ContentPart],
        schema: Type[BaseModel],
    ) -> str: ...


# ---------------------------------------------------------------------------
# Gemini backend (google-genai)
# ---------------------------------------------------------------------------


class GeminiBackend:
    def __init__(self, api_key: str):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)

    def _to_parts(self, contents: list[ContentPart]) -> list[Any]:
        from google.genai import types
        parts: list[Any] = []
        for c in contents:
            if isinstance(c, TextPart):
                parts.append(types.Part(text=c.text))
            else:
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=_pil_to_png_bytes(c.image),
                    )
                ))
        return parts

    def generate_image(
        self, model: str, contents: list[ContentPart], timeout_seconds: int
    ) -> Optional[PIL.Image.Image]:
        from google.genai import types
        config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000)
        )
        response = self.client.models.generate_content(
            model=model, contents=self._to_parts(contents), config=config
        )
        return _extract_gemini_image(response)

    def generate_json(
        self,
        model: str,
        contents: list[ContentPart],
        schema: Type[BaseModel],
    ) -> str:
        from google.genai import types
        # Gemini takes text + PIL.Image directly when contents is a flat list.
        flat: list[Any] = []
        for c in contents:
            flat.append(c.text if isinstance(c, TextPart) else c.image)
        response = self.client.models.generate_content(
            model=model,
            contents=flat,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return response.text


def _extract_gemini_image(response: Any) -> Optional[PIL.Image.Image]:
    parts = []
    if getattr(response, "parts", None):
        parts = response.parts
    elif response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts or []
    for part in parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return PIL.Image.open(io.BytesIO(part.inline_data.data))
    return None


# ---------------------------------------------------------------------------
# OpenRouter backend (OpenAI SDK)
# ---------------------------------------------------------------------------


class OpenRouterBackend:
    """Talks to OpenRouter's OpenAI-compatible REST API directly via httpx."""

    def __init__(self, api_key: str, base_url: str, gen_config: Optional[dict] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(60.0, read=300.0))
        self.gen_config = gen_config or {}

    def _params_for(self, model: str) -> dict:
        """Resolve gen-config sampling params for a model.

        A ``null`` value in ``per_model`` opts a key out of the default —
        useful for image models that 400 on `temperature` or `seed`.
        """
        merged = dict(self.gen_config.get("default") or {})
        merged.update(self.gen_config.get("per_model", {}).get(model) or {})
        return {k: v for k, v in merged.items() if v is not None}

    def _post_chat(self, payload: dict, timeout_seconds: int) -> dict:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(timeout_seconds, read=timeout_seconds),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    @staticmethod
    def _content_array(contents: list[ContentPart]) -> list[dict]:
        out: list[dict] = []
        for c in contents:
            if isinstance(c, TextPart):
                if c.text:
                    out.append({"type": "text", "text": c.text})
            else:
                out.append({
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(c.image)},
                })
        return out

    def generate_image(
        self, model: str, contents: list[ContentPart], timeout_seconds: int
    ) -> Optional[PIL.Image.Image]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self._content_array(contents)}],
            "modalities": ["image", "text"],
        }
        payload.update(_resolution_params(model))
        payload.update(self._params_for(model))
        return _extract_openrouter_image(self._post_chat(payload, timeout_seconds))

    def generate_json(
        self,
        model: str,
        contents: list[ContentPart],
        schema: Type[BaseModel],
    ) -> str:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system = (
            "You must reply with a single JSON object that conforms to this schema. "
            "Output JSON only, no prose, no code fences.\n\n"
            f"Schema:\n{schema_json}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._content_array(contents)},
            ],
            "response_format": {"type": "json_object"},
        }
        payload.update(self._params_for(model))
        body = self._post_chat(payload, timeout_seconds=300)
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return _strip_to_json(text)


def _resolution_params(model: str) -> dict:
    """Provider-specific request params to pin output to 1024x1024.

    OpenRouter forwards these to the provider; unknown keys are ignored,
    but we still scope by prefix so we never send an OpenAI param to a
    Gemini model (or vice versa) and risk a provider-side reject.
    """
    if model.startswith("google/"):
        return {"image_config": {"aspect_ratio": "1:1", "image_size": "1K"}}
    if model.startswith("openai/"):
        return {"size": "1024x1024"}
    return {}


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_to_json(text: str) -> str:
    """Some models still wrap JSON in code fences despite json_object — peel them."""
    cleaned = _FENCE_RE.sub("", text).strip()
    return cleaned


def _extract_openrouter_image(body: dict) -> Optional[PIL.Image.Image]:
    """Pull the first image out of an OpenRouter chat-completion JSON body.

    Different image-gen models on OpenRouter return images in slightly different
    shapes. We accept the common ones:
      - message.images: list of {"image_url": {"url": "data:..."}} (Google models)
      - message.images: list of base64 strings
      - content array with {"type": "image_url", "image_url": {"url": "..."}}
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}

    for entry in msg.get("images") or []:
        img = _decode_image_field(entry)
        if img is not None:
            return img

    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                img = _decode_data_url(url) if url else None
                if img is not None:
                    return img
    return None


def _decode_image_field(entry: Any) -> Optional[PIL.Image.Image]:
    if isinstance(entry, str):
        return _decode_data_url(entry) or _decode_b64(entry)
    if isinstance(entry, dict):
        url = (entry.get("image_url") or {}).get("url") if "image_url" in entry else None
        if url:
            return _decode_data_url(url)
        b64 = entry.get("b64_json") or entry.get("data")
        if b64:
            return _decode_b64(b64)
    return None


def _decode_data_url(url: str) -> Optional[PIL.Image.Image]:
    if not url:
        return None
    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return _decode_b64(payload)
    return None


def _decode_b64(payload: str) -> Optional[PIL.Image.Image]:
    try:
        return PIL.Image.open(io.BytesIO(base64.b64decode(payload)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class Solver:
    """Generates an image of reactant X."""

    def __init__(
        self,
        backend: Backend,
        model: str,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def run(self, contents: list[ContentPart]) -> Optional[PIL.Image.Image]:
        for attempt in range(self.max_retries):
            try:
                return self.backend.generate_image(self.model, contents, self.timeout_seconds)
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                print(f"[solver] attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        return None


class Validator:
    """Checks whether predicted X plausibly transforms to C under rule A→B."""

    def __init__(self, backend: Backend, model: str, prompt: str):
        self.backend = backend
        self.model = model
        self.prompt = prompt

    def run(
        self,
        image_a: PIL.Image.Image,
        image_b: PIL.Image.Image,
        image_c: PIL.Image.Image,
        image_x: PIL.Image.Image,
    ) -> ValidationResult:
        contents: list[ContentPart] = [
            TextPart(self.prompt),
            ImagePart(image_a),
            ImagePart(image_b),
            ImagePart(image_c),
            ImagePart(image_x),
        ]
        text = self.backend.generate_json(self.model, contents, ValidationResult)
        return ValidationResult.model_validate_json(text)


class Verifier:
    """Compares predicted X with reference D for evaluation."""

    def __init__(self, backend: Backend, model: str, prompt: str):
        self.backend = backend
        self.model = model
        self.prompt = prompt

    def run(
        self,
        predicted: PIL.Image.Image,
        reference: PIL.Image.Image,
    ) -> VerificationResult:
        contents: list[ContentPart] = [
            TextPart(self.prompt),
            ImagePart(predicted),
            ImagePart(reference),
        ]
        text = self.backend.generate_json(self.model, contents, VerificationResult)
        return VerificationResult.model_validate_json(text)


# ---------------------------------------------------------------------------
# Backwards-compatible helper retained for any external callers.
# ---------------------------------------------------------------------------


def pil_to_part(image: PIL.Image.Image) -> ImagePart:
    return ImagePart(image)
