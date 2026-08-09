"""Image-generation dispatch — the ``Target`` for images.

Mirrors :mod:`uniinfer.completion` (``Target`` for chat): binds a
``provider@model`` + API key, owns provider resolution + the per-provider image
dialect + the upstream fetch + response shaping behind one small interface.
The HTTP router becomes a thin adapter.

The dialect seam (pollinations GET vs the generic OpenAI-compatible
``/images/generations`` POST) is a *real* seam — two adapters — so it lives
behind the interface, private to this module. Adding a third image dialect is a
new adapter, not another router branch.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import httpx

from uniinfer.completion import parse_provider_model
from uniinfer.factory import ProviderFactory


@dataclass
class ImageData:
    """One generated image. ``b64_json`` is always populated — when an upstream
    returns only a URL, the bytes are fetched and base64-encoded so callers
    never have to handle the two shapes."""

    b64_json: str
    url: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"b64_json": self.b64_json}
        if self.url:
            out["url"] = self.url
        return out


class ImageGenerationError(Exception):
    """An upstream image API failure, carrying the HTTP status + body so the
    router can map it to an ``HTTPException`` without inspecting dialect internals."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(body)


class _ImageDialect:
    """Internal seam: how a provider's image API is called and parsed."""

    async def fetch(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        n: int,
        size: str,
        api_key: Optional[str],
    ) -> list[ImageData]:
        raise NotImplementedError


def _parse_size(size: str) -> tuple[int, int]:
    try:
        if isinstance(size, str) and "x" in size:
            w, h = size.split("x", 1)
            return int(w), int(h)
    except Exception:
        pass
    return 512, 512


async def _item_to_image_data(client: httpx.AsyncClient, item: dict) -> ImageData:
    """Normalize one upstream response item to ImageData (resolve url→b64)."""
    b64 = item.get("b64_json")
    url = item.get("url")
    if b64:
        return ImageData(b64_json=b64, url=url)
    if url:
        resp = await client.get(url, timeout=60)
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return ImageData(b64_json=b64, url=url)
    return ImageData(b64_json="")


class _GenericImageDialect(_ImageDialect):
    """POST ``{base_url}/images/generations`` — the OpenAI-compatible shape used
    by tu/aqueduct, openai, openrouter, kilo, stepfun, … (tu collapses in: its
    base_url resolves to the aqueduct endpoint). Streams the upstream body for
    bounded memory; aclose()s deterministically via ``async with``."""

    def __init__(self, base_url: str):
        self.endpoint = base_url.rstrip("/") + "/images/generations"

    async def fetch(self, client, model, prompt, n, size, api_key):
        async with client.stream(
            "POST",
            self.endpoint,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model, "prompt": prompt, "n": n, "size": size},
        ) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise ImageGenerationError(resp.status_code, err)
            raw = await resp.aread()
        data = json.loads(raw)
        del raw
        items = [await _item_to_image_data(client, item) for item in data.get("data", [])]
        del data
        return items


class _PollinationsDialect(_ImageDialect):
    """GET ``https://gen.pollinations.ai/image/{prompt}?…`` — raw image bytes,
    one request per image. Anonymous-capable (api_key optional)."""

    _BASE = "https://gen.pollinations.ai/image"

    async def fetch(self, client, model, prompt, n, size, api_key):
        enc = urllib.parse.quote(prompt)
        width, height = _parse_size(size)
        headers = {"Accept": "image/jpeg", "User-Agent": "UniIOAI/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        out: list[ImageData] = []
        for i in range(n):
            seed = int(time.time()) + i
            url = f"{self._BASE}/{enc}?model={model}&width={width}&height={height}&seed={seed}"
            resp = await client.get(url, headers=headers, timeout=60)
            if resp.status_code != 200:
                raise ImageGenerationError(
                    resp.status_code, resp.text or "Failed to generate image from Pollinations"
                )
            out.append(ImageData(b64_json=base64.b64encode(resp.content).decode("utf-8"), url=url))
            del resp
        return out


def _provider_image_base_url(provider_name: str) -> Optional[str]:
    """The provider's image-API base URL from its class (BASE_URL / _DEFAULT_BASE_URL)."""
    try:
        cls = ProviderFactory._resolve(provider_name)
        return getattr(cls, "BASE_URL", None) or getattr(cls, "_DEFAULT_BASE_URL", None)
    except Exception:
        return None


class ImageTarget:
    """Bind a ``provider@model`` + key and own the image-generation dispatch.

    The single home for "generate an image" — the image analog of
    :class:`uniinfer.completion.Target`. Resolves the provider + its image
    dialect at construction; the caller injects an ``httpx.AsyncClient`` at call
    time (dependency injection for testability).

    Args:
        provider_model: ``provider@model`` (e.g. ``tu@z-image-turbo``).
        api_key: The provider API key (None for anonymous providers like pollinations).

    Raises:
        ValueError: provider unknown, or no image base_url resolvable.
    """

    def __init__(self, provider_model: str, api_key: Optional[str] = None):
        self.provider_model = provider_model
        self.api_key = api_key
        self.provider_name, self.model_name = parse_provider_model(provider_model)
        self._dialect = self._resolve_dialect()

    def _resolve_dialect(self) -> _ImageDialect:
        if self.provider_name == "pollinations":
            return _PollinationsDialect()
        base = _provider_image_base_url(self.provider_name)
        if not base:
            raise ValueError(
                f"Image generation not supported for provider '{self.provider_name}'"
            )
        return _GenericImageDialect(base)

    async def agenerate(
        self,
        prompt: str,
        *,
        n: int = 1,
        size: str = "512x512",
        client: httpx.AsyncClient,
    ) -> list[ImageData]:
        """Generate ``n`` images. The client is injected (tests pass a fake).

        Raises:
            PermissionError: a key is required but none was provided.
            ImageGenerationError: the upstream image API returned non-200.
        """
        if not self.api_key and self.provider_name != "pollinations":
            raise PermissionError(f"API key required for {self.provider_name}")
        return await self._dialect.fetch(
            client, self.model_name, prompt, n, size, self.api_key
        )
