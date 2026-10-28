"""Selects an OpenAI-compatible chat provider.

OpenAI is the default. Set ``LLM_PROVIDER=deepseek`` and ``DEEPSEEK_API_KEY``
to send the same chat-completions calls to DeepSeek.
"""

import os
from typing import Literal

from openai import OpenAI

from virtual_lab.constants import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL

Provider = Literal["openai", "deepseek"]

_DEEPSEEK_PLACEHOLDER_KEYS = {"", "your_deepseek_api_key_here"}


def get_provider() -> Provider:
    """Returns the configured chat provider.

    ``LLM_PROVIDER`` may be ``openai`` (default) or ``deepseek``.
    """
    raw = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    if raw in {"", "openai"}:
        return "openai"
    if raw == "deepseek":
        return "deepseek"
    raise ValueError(f"Unknown LLM_PROVIDER {raw!r}. Use 'openai' or 'deepseek'.")


def create_client() -> OpenAI:
    """Creates a chat client for the configured provider.

    DeepSeek uses the OpenAI SDK with ``base_url`` set to the DeepSeek API.
    """
    if get_provider() == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if api_key in _DEEPSEEK_PLACEHOLDER_KEYS:
            raise RuntimeError(
                "LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY. "
                "Get a key at https://platform.deepseek.com/api_keys"
            )
        base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).strip()
        return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_DEEPSEEK_BASE_URL)
    return OpenAI()


def resolve_model(model: str) -> str:
    """Returns the model name to send to the configured provider.

    DeepSeek cannot serve OpenAI model ids (``gpt-*``, ``o1-*``, ``ft:*``).
    Those names are replaced by ``DEEPSEEK_MODEL`` (default ``deepseek-flash``).
    A model id that already starts with ``deepseek`` is left unchanged, so
    individual agents can still pin ``deepseek-flash`` or ``deepseek-v4-pro``.
    """
    if get_provider() != "deepseek" or model.startswith("deepseek"):
        return model
    override = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
    return override or DEFAULT_DEEPSEEK_MODEL


def deepseek_thinking_enabled() -> bool:
    """Whether DeepSeek thinking mode is turned on.

    DeepSeek enables thinking by default, which ignores ``temperature``.
    This package disables it unless ``DEEPSEEK_THINKING`` is set to enabled.
    """
    raw = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"", "0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(
        f"Unknown DEEPSEEK_THINKING {raw!r}. Use 'enabled' or 'disabled'."
    )


def chat_completion_kwargs(temperature: float) -> dict[str, object]:
    """Returns provider-specific arguments for ``chat.completions.create``.

    Sampling temperature is sent for OpenAI and for DeepSeek non-thinking mode.
    Thinking mode ignores temperature, so it is omitted there.
    """
    if get_provider() != "deepseek":
        return {"temperature": temperature}
    if deepseek_thinking_enabled():
        return {"extra_body": {"thinking": {"type": "enabled"}}}
    return {
        "temperature": temperature,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
