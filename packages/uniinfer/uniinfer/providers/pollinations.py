from __future__ import annotations
"""
Pollinations provider implementation.

Uses the rich ``gen.pollinations.ai/models`` endpoint (NOT the bare ``/v1/models``,
which carries no pricing). The rich endpoint exposes per-model ``pricing`` in
**pollen** (Pollinations' credit currency), plus brand, title, description,
capabilities, and modalities.

Access is derived from pricing — **no pollen is consumed** (read-only catalog
fetch): a model whose pricing has no positive token/image cost field is FREE
(the ``∞`` tier on enter.pollinations.ai/models); any positive cost field means
PAID. This is the reliable signal — unlike a probe, which spends pollen, and
unlike cost-zero heuristics elsewhere (pollinations expresses free as the
*absence* of cost fields, not a zero value).

``cost`` (USD per 1M tokens, the ModelInfo convention): only genuinely-free
models get ``{input: 0, output: 0}``. Paid models keep their pollen pricing in
``raw`` only — pollinations publishes no pollen→USD rate, and writing pollen
values into a USD-priced field would silently misprice them downstream
(pi's models.json, model pickers). ``access: "paid"`` still marks them.
"""
from decimal import Decimal, InvalidOperation
from typing import Optional

import requests

from ..errors import map_provider_error
from .openai_compatible import OpenAICompatibleChatProvider


def _pollinations_model_type(in_mods: list, out_mods: list) -> str:
    """Classify by output modality first, then input.

    The output modality is the strong signal: a model that *emits* video/audio/
    image is not a chat model no matter what its id looks like. This is what
    kept video models (e.g. nova-reel) and STT models out of the chat pool.
    """
    if "video" in out_mods:
        return "video"
    if "audio" in out_mods:
        return "tts"
    if "image" in out_mods:
        return "image"
    if set(in_mods) <= {"audio"} and set(out_mods) <= {"text"}:
        return "stt"
    return "chat"


class PollinationsProvider(OpenAICompatibleChatProvider):
    """Provider for the Pollinations OpenAI-compatible API."""

    BASE_URL = "https://gen.pollinations.ai/v1"
    PROVIDER_ID = "pollinations"
    ERROR_PROVIDER_NAME = "Pollinations"
    DEFAULT_MODEL = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list["ModelInfo"]:
        """List models from Pollinations via the rich ``/models`` endpoint.

        ``gen.pollinations.ai/models`` carries per-model ``pricing`` (pollen),
        brand, description, capabilities + modalities — unlike the bare
        ``/v1/models``. Access: no positive token/image cost field → FREE
        (the ``∞`` tier); else PAID. Read-only — no pollen spent.
        """
        from ..core import ModelInfo

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        def _positive(pricing: dict, field: str) -> bool:
            v = pricing.get(field)
            if v is None:
                return False
            try:
                return Decimal(str(v)) > 0
            except (InvalidOperation, ValueError):
                return False

        def _is_free(m: dict) -> bool:
            p = m.get("pricing") or {}
            return not any(
                _positive(p, f)
                for f in ("promptTextTokens", "completionTextTokens", "completionImageTokens")
            )

        def _cost(m: dict, is_free: bool) -> dict | None:
            """USD per 1M tokens — only when we actually know it.

            Free models are $0 in any currency. Paid models are priced in
            pollen; with no published pollen→USD rate we report no cost
            (rather than mislabel pollen as USD) — access still says "paid".
            """
            if not is_free:
                return None
            return {"input": 0, "output": 0}

        try:
            r = requests.get("https://gen.pollinations.ai/models", headers=headers, timeout=30)
            if r.status_code != 200:
                raise map_provider_error(
                    "Pollinations",
                    Exception(f"Pollinations API error: {r.status_code} - {r.text}"),
                    status_code=r.status_code,
                    response_body=r.text,
                )
            data = r.json()
            models = data.get("data", data) if isinstance(data, dict) else data
        except Exception as e:
            raise map_provider_error("Pollinations", e)

        results: list[ModelInfo] = []
        for m in models:
            if not isinstance(m, dict):
                continue
            mid = m.get("name") or m.get("id")
            if not mid:
                continue
            caps_list = m.get("capabilities") or []
            caps: dict = {}
            if any(c in ("tool_calling", "tools") for c in caps_list):
                caps["tool_call"] = True
            if "reasoning" in caps_list:
                caps["reasoning"] = True
            in_mods = m.get("input_modalities") or ["text"]
            out_mods = m.get("output_modalities") or ["text"]
            if "image" in in_mods:
                caps["vision"] = True
            is_free = _is_free(m)
            results.append(ModelInfo(
                id=mid,
                name=m.get("title") or mid,
                type=_pollinations_model_type(in_mods, out_mods),
                context_window=m.get("context_length"),
                access="free" if is_free else "paid",
                cost=_cost(m, is_free),
                modalities={"input": in_mods, "output": out_mods},
                capabilities=caps or None,
                owned_by=m.get("brand"),
                raw=m,
            ))
        return results
