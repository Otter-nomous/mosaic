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
import concurrent.futures
import io
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Type, Union

import httpx
import PIL.Image
from pydantic import BaseModel

from .schemas import ValidationResult, VerificationResult


class OpenRouterHTTPError(RuntimeError):
    """HTTP error from OpenRouter; ``status_code`` is the upstream HTTP status."""

    def __init__(self, status_code: int, body: str, retry_after: Optional[float] = None):
        super().__init__(f"OpenRouter HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Neutral content representation (provider-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class TextPart:
    text: str
    cache_breakpoint: bool = False


@dataclass
class ImagePart:
    image: PIL.Image.Image
    cache_breakpoint: bool = False


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


_GEMINI_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "seed"}


class GeminiBackend:
    def __init__(self, api_key: str, gen_config: Optional[dict] = None):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        # Same shape as OpenRouterBackend.gen_config:
        #   {"default": {...}, "per_model": {model: {...}}}
        # A ``null`` value in per_model opts that key out of the default —
        # useful for image-gen variants that 400 on `temperature` or `seed`.
        self.gen_config = gen_config or {}

    def _params_for(self, model: str) -> dict:
        merged = dict(self.gen_config.get("default") or {})
        merged.update(self.gen_config.get("per_model", {}).get(model) or {})
        return {
            k: v for k, v in merged.items()
            if v is not None and k in _GEMINI_SAMPLING_KEYS
        }

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
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
            **self._params_for(model),
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
                **self._params_for(model),
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

    def __init__(
        self,
        api_key: str,
        base_url: str,
        gen_config: Optional[dict] = None,
        max_concurrent_calls: int = 4,
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(60.0, read=300.0))
        self.gen_config = gen_config or {}
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        # Wall-clock cap on each call: httpx's read timeout is per-byte, so a
        # slow-streaming upstream can keep a connection alive past it. Run each
        # request in a worker thread and use Future.result(timeout=...) to bound
        # total elapsed time per attempt. ``max_concurrent_calls`` should be at
        # least the caller's parallel agent count, or calls will queue here
        # before reaching httpx.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, max_concurrent_calls), thread_name_prefix="orb"
        )

    def _params_for(self, model: str) -> dict:
        """Resolve gen-config sampling params for a model.

        A ``null`` value in ``per_model`` opts a key out of the default —
        useful for image models that 400 on `temperature` or `seed`.
        """
        merged = dict(self.gen_config.get("default") or {})
        merged.update(self.gen_config.get("per_model", {}).get(model) or {})
        return {k: v for k, v in merged.items() if v is not None}

    def _post_chat(self, payload: dict, timeout_seconds: int) -> dict:
        """POST /chat/completions with backoff for 429 + 5xx and timeouts.

        Retries:
          - HTTP 429 (rate-limited): honor ``Retry-After`` header if present,
            else exponential backoff with jitter
          - HTTP 5xx: same exponential backoff
          - httpx connect/read timeouts: same exponential backoff

        4xx other than 429 is not retried — those signal a malformed request.
        """
        def _do() -> dict:
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
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                raise OpenRouterHTTPError(
                    response.status_code, response.text, retry_after
                )
            return response.json()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                future = self._executor.submit(_do)
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                last_exc = httpx.ReadTimeout(
                    f"wall-clock cap of {timeout_seconds}s exceeded"
                )
                last_exc.__cause__ = exc
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                    httpx.RemoteProtocolError) as exc:
                last_exc = exc
            except OpenRouterHTTPError as exc:
                # Only retry rate-limit (429) and server errors (5xx).
                if exc.status_code != 429 and not (500 <= exc.status_code < 600):
                    raise
                last_exc = exc

            if attempt >= self.max_retries:
                break
            wait = _compute_backoff(
                attempt,
                self.retry_base_seconds,
                self.retry_max_seconds,
                getattr(last_exc, "retry_after", None),
            )
            print(
                f"[openrouter] {type(last_exc).__name__} "
                f"({getattr(last_exc, 'status_code', '-')}); "
                f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _content_array(contents: list[ContentPart]) -> list[dict]:
        out: list[dict] = []
        for c in contents:
            if isinstance(c, TextPart):
                if not c.text:
                    continue
                part: dict = {"type": "text", "text": c.text}
            else:
                part = {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(c.image)},
                }
            if c.cache_breakpoint:
                part["cache_control"] = {"type": "ephemeral"}
            out.append(part)
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
        body = self._post_chat(payload, timeout_seconds=600)
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return _strip_to_json(text)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After`` header — only the seconds form (HTTP-date is rare)."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None


def _compute_backoff(
    attempt: int, base: float, cap: float, retry_after: Optional[float]
) -> float:
    """Exponential backoff with full jitter; preferred to ``Retry-After`` when set."""
    if retry_after is not None and retry_after > 0:
        # Add a small jitter so concurrent callers don't unblock in lockstep.
        return min(cap, retry_after + random.uniform(0.0, 1.0))
    upper = min(cap, base * (2 ** attempt))
    return random.uniform(base, upper)


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
# Backend factory (auto-route by model name)
# ---------------------------------------------------------------------------


def is_gemini_direct(model: str) -> bool:
    """A bare ``gemini-...`` name (no ``/``) routes to GeminiBackend.

    Slash-prefixed names (e.g. ``google/gemini-3-pro-image-preview``,
    ``openai/gpt-5.4``) route to OpenRouter. The two model-name conventions are
    already used throughout ``mosaic.models``.
    """
    return "/" not in model


def make_backend(
    model: str,
    *,
    openrouter_api_key: Optional[str],
    google_api_key: Optional[str],
    openrouter_base_url: str,
    openrouter_max_concurrent_calls: int = 4,
    gen_config: Optional[dict] = None,
) -> Backend:
    """Pick the right backend for ``model`` based on its name.

    Raises if the required key for the chosen provider is missing. ``gen_config``
    is the unified sampling-params config (same shape for both backends);
    OpenRouter-only keys (e.g. ``provider`` overrides) are ignored by
    GeminiBackend, and vice versa.
    """
    if is_gemini_direct(model):
        if not google_api_key:
            raise RuntimeError(
                f"model {model!r} routes to Gemini direct; set $GOOGLE_API_KEY "
                f"or pass --google-api-key-file."
            )
        return GeminiBackend(api_key=google_api_key, gen_config=gen_config)
    if not openrouter_api_key:
        raise RuntimeError(
            f"model {model!r} routes through OpenRouter; set $OPENROUTER_API_KEY "
            f"or pass --api-key-file."
        )
    return OpenRouterBackend(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
        max_concurrent_calls=openrouter_max_concurrent_calls,
        gen_config=gen_config,
    )


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
        timeout_seconds: int = 600,
    ):
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def run(self, contents: list[ContentPart]) -> Optional[PIL.Image.Image]:
        # The solver call has no dynamic suffix — entire content is static across
        # repeated trials of the same example. Mark the last part as the cache
        # breakpoint so the whole prefix is eligible for prompt caching.
        if contents:
            contents[-1].cache_breakpoint = True
        for attempt in range(self.max_retries):
            try:
                result = self.backend.generate_image(
                    self.model, contents, self.timeout_seconds
                )
                if result is not None:
                    return result
                # API returned 200 but the response had no image — model emitted
                # text (refusal/explanation) instead. Treat as retryable.
                raise RuntimeError("backend returned 200 with no image")
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
        # Static prefix [prompt, A, B, C] is identical across trials; X is dynamic.
        # Mark image_c so the prompt+A+B+C bytes become the cache prefix.
        contents: list[ContentPart] = [
            TextPart(self.prompt),
            ImagePart(image_a),
            ImagePart(image_b),
            ImagePart(image_c, cache_breakpoint=True),
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
