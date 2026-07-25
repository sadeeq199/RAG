"""OpenRouter prompting layer for context-grounded answers."""

from __future__ import annotations

import os
from typing import Any

import requests

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit is present in the app runtime.
    st = None  # type: ignore[assignment]

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - python-dotenv is a hard requirement.
    dotenv_values = None  # type: ignore[assignment]

_ENV_FILE_VALUES = dotenv_values(".env") if dotenv_values is not None else {}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
PROMPT_TEMPLATE = """You are a helpful AI assistant.

Answer ONLY using the retrieved context.

If the answer cannot be found inside the context,
reply:

"I cannot answer from the provided documents."

Context:

{context}

Question:

{question}

Answer:
"""


def _read_secret(name: str, default: str | None = None) -> str | None:
    """Read a config value with priority: st.secrets, then .env, then os.environ."""
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass

    env_file_value = _ENV_FILE_VALUES.get(name)
    if env_file_value:
        return env_file_value

    return os.getenv(name, default)


def get_openrouter_config() -> tuple[str, str]:
    """Return the configured OpenRouter API key and model name."""
    api_key = _read_secret("OPENROUTER_API_KEY")
    model = _read_secret("OPENROUTER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in Streamlit secrets or environment.")
    return api_key, model


def build_prompt(context: str, question: str) -> str:
    """Build the strict context-only RAG prompt."""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer_question(context: str, question: str, timeout: int = 60) -> str:
    """Generate an answer with OpenRouter using only retrieved context."""
    api_key, model = get_openrouter_config()
    prompt = build_prompt(context=context, question=question)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "RAG Document Assistant",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected response format.") from exc
