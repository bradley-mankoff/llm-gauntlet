"""Thin OpenAI SDK wrapper pointing at the local llama-server.

The server exposes an OpenAI-compatible API at /v1; this module just sets
sane defaults (api_key=noop, base_url from env) and gives a tiny helper.
"""
from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8080/v1"


def make_client(base_url: str | None = None, api_key: str = "sk-noop", timeout: float = 1800.0) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("LLAMA_API_KEY", api_key),
        timeout=timeout,
    )


def list_models(client: OpenAI) -> list[str]:
    return [m.id for m in client.models.list().data]


def chat(
    client: OpenAI,
    model: str,
    messages: Iterable[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    no_think: bool = False,
    think_off: bool = False,
    **kwargs,
):
    """Send a chat completion request.

    Two ways to disable Qwen 3.5/3.6 reasoning, in increasing order of forcefulness:

      no_think=True:   appends `/no_think` to the last user message (Qwen's
                       "soft switch" — the model *can* choose to honor it, gave
                       ~30% reduction in a 27B Q4_K_XL smoke test).

      think_off=True:  prepends `<|think_off|>` to the last user message.
                       This is the FROGGERIC CHAT TEMPLATE'S hard switch
                       (froggeric/Qwen-Fixed-Chat-Templates) — the template
                       strips the tag before the model sees it and flips
                       reasoning off internally. With the upstream Qwen
                       template (no froggeric), this tag is just a string the
                       model reads and ignores. Requires the server to be
                       launched with USE_FROGGERIC_CHAT_TEMPLATE=1.
                       Verified: drops completion_tokens 1316 -> 20 on the
                       same haiku prompt.

    Safe to call with both False (the default) on any model.
    """
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "user":
        content = msgs[-1].get("content", "")
        if think_off and "<|think_off|>" not in content:
            msgs[-1] = {**msgs[-1], "content": f"<|think_off|>\n{content}"}
        if no_think and "/no_think" not in content and "<|think_off|>" not in content:
            sep = "" if content.endswith((" ", "\n", "\t")) else " "
            msgs[-1] = {**msgs[-1], "content": f"{content}{sep}/no_think"}
    return client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


def message_text(message) -> str:
    """Best-effort assistant text across reasoning-only models.

    Some local templates put the whole reply in ``reasoning_content`` and leave
    ``content`` empty unless ``--reasoning off`` is set on the server.
    """
    content = getattr(message, "content", None) or ""
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        val = getattr(message, key, None)
        if isinstance(val, str) and val.strip():
            return val
    # openai SDK model_extra / dict forms
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return content if isinstance(content, str) else ""

