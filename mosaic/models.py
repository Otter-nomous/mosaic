"""Curated OpenRouter model lists for MOSAIC agents.

Solver models generate images. Validator/Verifier models are text/vision
reasoning models that emit structured JSON.
"""

from __future__ import annotations

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SOLVER_MODELS_OPENROUTER: list[str] = [
    "google/gemini-3-pro-image-preview",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-2.5-flash-image",
    "openai/gpt-5.4-image-2",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
]

VALIDATOR_MODELS_OPENROUTER: list[str] = [
    "moonshotai/kimi-k2.6",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-4.7",
    "x-ai/grok-4.1-fast",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "moonshotai/kimi-k2.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "qwen/qwen3.6-plus",
    "openai/gpt-5.5",
]

DEFAULT_SOLVER_OPENROUTER = "google/gemini-3-pro-image-preview"
DEFAULT_VALIDATOR_OPENROUTER = "google/gemini-3-flash-preview"
DEFAULT_VERIFIER_OPENROUTER = "google/gemini-3-flash-preview"

DEFAULT_SOLVER_GEMINI = "gemini-3-pro-image-preview"
DEFAULT_VALIDATOR_GEMINI = "gemini-3-flash-preview"
DEFAULT_VERIFIER_GEMINI = "gemini-3-flash-preview"
