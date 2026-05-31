from __future__ import annotations

import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_GEMMA_MODEL = "google/gemma-4-31B-it:novita"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(part for part in text_parts if part).strip()
    return ""


def request_chat_completion(
    messages: list[dict[str, Any]],
    hf_token: str,
    model: str = DEFAULT_GEMMA_MODEL,
    endpoint: str = DEFAULT_HF_ENDPOINT,
    timeout_seconds: int = 30,
    requester: Callable[..., Any] | None = None,
) -> str:
    """Call Hugging Face chat completions API and return assistant text output."""
    if not hf_token or not hf_token.strip():
        raise ValueError("Hugging Face token is required.")
    if not messages:
        raise ValueError("At least one message is required.")

    payload = {
        "messages": messages,
        "model": model,
        "stream": False,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + hf_token,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    requester = requester or urlopen

    try:
        with requester(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to reach Hugging Face API: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Hugging Face API returned invalid JSON.") from exc

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Hugging Face API response did not include choices.")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("Hugging Face API response did not include a valid message.")

    text = _extract_text_content(message.get("content"))
    if not text:
        raise RuntimeError("Hugging Face API response was empty.")

    return text
