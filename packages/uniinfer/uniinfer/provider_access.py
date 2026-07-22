"""
Proxy service-layer helpers: credgoo key resolution, embeddings, and model
listing.

Chat completion dispatch lived here historically (get_completion /
stream_completion / aget_completion / astream_completion) but has moved to
``uniinfer.completion.Target`` — the deep module that owns parse ->
instantiate -> request-build -> dispatch -> access-recording. This module now
holds only the concerns that are NOT completion dispatch: API-key resolution,
embeddings (different factory + request type), and the list helpers.
"""
import logging
import os
from typing import Any

from uniinfer import EmbeddingProviderFactory, EmbeddingRequest, EmbeddingResponse
from uniinfer import ProviderFactory
from uniinfer.completion import parse_provider_model
from uniinfer.config.providers import PROVIDER_CONFIGS
from uniinfer.errors import AuthenticationError, UniInferError
from uniinfer.json_utils import update_models

from credgoo import get_api_key
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = os.path.join(os.getcwd(), ".env")
found_dotenv = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
logger.debug(f"Attempted to load .env from: {dotenv_path}")
logger.debug(f".env file found and loaded: {found_dotenv}")


def _resolve_credgoo_service(provider_name: str) -> str:
    """Resolve the credgoo service name for a provider.

    Prefers the provider class's CREDGOO_SERVICE attribute (e.g. the kilo
    gateway stores its key under the 'kilocode' service); falls back to the
    provider id for providers that don't override it. Mirrors the CLI resolver
    in uniinfer_cli._resolve_credgoo_service.
    """
    try:
        from uniinfer import ProviderFactory
        cls = ProviderFactory.get_provider_class(provider_name)
        credgoo_service = getattr(cls, "CREDGOO_SERVICE", None)
        if credgoo_service:
            return credgoo_service
    except Exception:
        pass
    return provider_name


# --- API Key Retrieval ---


def get_provider_api_key(api_bearer_token: str, provider_name: str) -> str | None:
    """Resolve a provider API key from a direct key or a credgoo combined token.

    Args:
        api_bearer_token: A direct provider API key, a combined credgoo token
            ('bearer@encryption'), or None/empty for providers like Ollama.
        provider_name: The provider name (e.g. 'openai', 'ollama').

    Returns:
        The resolved provider API key (None for providers like Ollama).

    Raises:
        ValueError: If the combined credgoo token format is invalid or a key
            is required but missing.
        AuthenticationError: If credgoo fails to retrieve a key.
    """
    provider_api_key = None

    # Fallback to env vars if token is missing
    if not api_bearer_token:
        files_token = os.getenv("CREDGOO_BEARER_TOKEN")
        files_enc = os.getenv("CREDGOO_ENCRYPTION_KEY")
        if files_token and files_enc:
            api_bearer_token = f"{files_token}@{files_enc}"

    if api_bearer_token and "@" in api_bearer_token:
        # Treat as credgoo combined token: bearer@encryption
        try:
            credgoo_bearer, credgoo_encryption = api_bearer_token.split("@", 1)
            if not credgoo_bearer or not credgoo_encryption:
                raise ValueError(
                    "Invalid combined credgoo token format. Both parts are required."
                )
            credgoo_service = _resolve_credgoo_service(provider_name)
            provider_api_key = get_api_key(
                service=credgoo_service,
                encryption_key=credgoo_encryption,
                bearer_token=credgoo_bearer,
            )
            if not provider_api_key and provider_name not in ["ollama"]:
                raise AuthenticationError(
                    f"Failed to retrieve API key for '{provider_name}' using the provided credgoo token."
                )
        except ValueError as e:
            raise ValueError(
                f"Invalid combined credgoo token format ('bearer@encryption'): {e}"
            )
        except Exception as e:
            raise AuthenticationError(
                f"Error retrieving key from credgoo for '{provider_name}': {e}"
            )
    else:
        if api_bearer_token:
            provider_api_key = api_bearer_token
        elif provider_name != "ollama":
            raise ValueError(
                "API Bearer Token is required (provider key or credgoo combo)."
            )

    if not provider_api_key and provider_name not in ["ollama"]:
        logger.warning(
            f"API key for {provider_name} is missing or empty after processing."
        )

    return provider_api_key


# --- Embeddings (separate from completion: different factory + request type) ---


def get_embeddings(
    input_texts: list[str],
    provider_model_string: str,
    provider_api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run an embedding request via uniinfer.

    Args:
        input_texts: Texts to embed.
        provider_model_string: ``provider@model`` (e.g. 'ollama@nomic-embed-text:latest').
        provider_api_key: Pre-resolved provider API key (None for Ollama).
        base_url: Optional provider base URL.

    Returns:
        ``{'embeddings': [...], 'usage': {...}}``.

    Raises:
        ValueError: If provider_model_string is malformed.
        UniInferError: On provider/request failure.
    """
    try:
        provider_name, model_name = parse_provider_model(provider_model_string)

        provider_kwargs: dict[str, Any] = {"api_key": provider_api_key}
        if base_url:
            provider_kwargs["base_url"] = base_url

        provider = EmbeddingProviderFactory.get_provider(provider_name, **provider_kwargs)
        request = EmbeddingRequest(input=input_texts, model=model_name)

        logger.info(f"Requesting embeddings from {provider_name} ({model_name})")
        response: EmbeddingResponse = provider.embed(request)
        logger.info("Embeddings received")

        embeddings = [ed["embedding"] for ed in response.data]
        usage = getattr(response, "usage", {"prompt_tokens": 0, "total_tokens": 0})
        return {"embeddings": embeddings, "usage": usage}

    except (UniInferError, ValueError) as e:
        logger.error(f"An error occurred during embedding request: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during embedding request: {e}")
        raise UniInferError(f"An unexpected error occurred in get_embeddings: {e}")


# --- Listing helpers ---


def list_embedding_providers() -> list[str]:
    """Return the names of all available embedding providers."""
    return EmbeddingProviderFactory.list_providers()


def list_embedding_models_for_provider(
    provider_name: str, api_bearer_token: str
) -> list[str]:
    """Return available embedding model names for the given provider.

    For Ollama, api_bearer_token can be empty or None. Updates the models.json cache.
    """
    if provider_name == "ollama":
        api_key = None
    else:
        api_key = get_provider_api_key(api_bearer_token, provider_name)
    extra: dict[str, Any] = {}
    if provider_name in ["cloudflare", "ollama"]:
        extra = PROVIDER_CONFIGS.get(provider_name, {}).get("extra_params", {})
    provider_cls = EmbeddingProviderFactory.get_provider_class(provider_name)
    modellist = provider_cls.list_models(api_key=api_key, **extra)
    from uniinfer.proxy_services.models_registry import Catalog

    Catalog().upsert_provider(provider_name, modellist)
    return modellist


def list_providers() -> list[str]:
    """Return the names of all available providers."""
    return ProviderFactory.list_providers()


def list_models_for_provider(provider_name: str, api_bearer_token: str) -> list[str]:
    """Return available model names for the given provider.

    Uses the bearer token to resolve the API key. Updates models.json cache for
    subsequent /v1/models calls.
    """
    api_key = get_provider_api_key(api_bearer_token, provider_name)
    extra: dict[str, Any] = {}
    if provider_name in ["cloudflare", "ollama"]:
        extra = PROVIDER_CONFIGS.get(provider_name, {}).get("extra_params", {})
    provider_cls = ProviderFactory.get_provider_class(provider_name)
    modellist = provider_cls.list_models(api_key=api_key, **extra)
    update_models(modellist, provider_name)
    from uniinfer.proxy_services.models_registry import Catalog

    Catalog().upsert_provider(provider_name, modellist)
    return modellist


def list_model_names_for_provider(
    provider_name: str, api_bearer_token: str
) -> list[str]:
    """Return model ID strings for the given provider.

    Convenience wrapper around list_models_for_provider that always returns
    plain strings regardless of whether the provider returns ModelInfo objects.
    """
    modellist = list_models_for_provider(provider_name, api_bearer_token)
    return [m.id if hasattr(m, "id") else str(m) for m in modellist]
