"""Response extraction utilities for ChatCompletionResponse.

Handles thinking models (Qwen 3.x, DeepSeek) that put content in
``reasoning_content`` instead of ``content``, and providers that expose
a separate ``thinking`` attribute.
"""


def extract_response_text(response) -> str:
    """Extract text from a ChatCompletionResponse, handling thinking models.

    Tries (first non-empty wins):
    1. ``response.message.content``
    2. ``response.raw_response["choices"][0]["message"]["content"]``
    3. ``response.raw_response["choices"][0]["message"]["reasoning_content"]``
    4. ``response.thinking``

    Returns ``""`` if no text could be extracted.
    """
    # 1. Primary: message.content
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    # 2-3. Raw response fallback (thinking models put text in reasoning_content)
    raw = getattr(response, "raw_response", None)
    if isinstance(raw, dict):
        try:
            choices = raw.get("choices") or []
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                for key in ("content", "reasoning_content"):
                    val = msg.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        except Exception:
            pass

    # 4. thinking attribute (some providers expose it directly)
    thinking = getattr(response, "thinking", None)
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()

    return ""
