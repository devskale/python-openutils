import json
import os
import re
from datetime import datetime
from typing import Any

_REDACT_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "bearer",
)

_REDACT_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)sk-[a-z0-9]{20,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)api[_-]?key\s*[:=]\s*[a-z0-9._\-]+"), "api_key=[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\+?\d{1,3}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def _redact_string(value: str) -> str:
    redacted = value
    for pattern, repl in _REDACT_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    return redacted


def redact_data(data: Any) -> Any:
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in _REDACT_KEYS):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_data(value)
        return result
    if isinstance(data, list):
        return [redact_data(item) for item in data]
    if isinstance(data, str):
        return _redact_string(data)
    return data


def log_raw_response(
    provider: str,
    operation: str,
    raw_response: Any,
    log_file: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    log_dir = os.getenv("UNIINFER_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(log_dir, exist_ok=True)

    log_path = log_file or os.path.join(log_dir, "provider_raw_responses.log")
    event: dict[str, Any] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "provider": provider,
        "operation": operation,
        "data": redact_data(raw_response),
    }
    if extra:
        event.update(redact_data(extra))

    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    except (OSError, PermissionError):
        return
