from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaChatResult:
    content: str
    model: str
    total_duration_ns: int | None = None


def chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_format: str = "json",
    timeout_seconds: float = 120.0,
) -> OllamaChatResult:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if response_format == "json":
        payload["format"] = "json"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise ConnectionError(f"Failed to reach Ollama at {url}: {error}") from error

    message = raw.get("message", {})
    return OllamaChatResult(
        content=message.get("content", ""),
        model=raw.get("model", model),
        total_duration_ns=raw.get("total_duration"),
    )


def extract_json_object(content: str) -> dict:
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON object from model response: {cleaned[:200]}")
