"""Groq client wrapper for the Cascade Bank local-first RAG demo."""

from __future__ import annotations

from typing import Any, Iterable

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_config, require_groq_key

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Build and cache an OpenAI-compatible client pointed at Groq."""
    global _client
    if _client is None:
        cfg = get_config()
        _client = OpenAI(
            base_url=cfg.generation.base_url,
            api_key=require_groq_key(),
        )
    return _client


_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _create_completion(client: OpenAI, **kwargs: Any):
    return client.chat.completions.create(**kwargs)


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> str | Iterable:
    """Call Groq chat completions, using configured defaults when omitted."""
    cfg = get_config()
    client = get_client()
    kwargs = {
        "model": model or cfg.generation.model,
        "messages": messages,
        "max_tokens": cfg.generation.max_tokens if max_tokens is None else max_tokens,
        "stream": stream,
    }
    resolved_model = kwargs["model"]
    if str(resolved_model).startswith("openai/gpt-oss-"):
        # GPT-OSS uses part of the completion budget for internal reasoning.
        # Keep that bounded so the user-facing answer is not truncated.
        kwargs["reasoning_effort"] = "medium"
    else:
        kwargs["temperature"] = (
            cfg.generation.temperature if temperature is None else temperature
        )
    if stream:
        return _create_completion(client, **kwargs)
    response = _create_completion(client, **kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(
            "Groq returned no user-facing answer. Increase generation.max_tokens "
            "if the response ended before answer generation."
        )
    return content
