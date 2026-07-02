"""
Provider factory for managing and instantiating chat providers.
"""
from typing import Dict, Type, Optional
from .core import ChatProvider

# Try to import credgoo for API key management


class ProviderFactory:
    """
    Factory for creating provider instances.
    """
    _providers: Dict[str, Type[ChatProvider]] = {}
    # Lazy registrations: name -> "dotted.module.path:ClassName". The module
    # is imported only on first use, so heavy provider SDKs (e.g. google-genai,
    # ~50s to import) don't slow startup unless that provider is actually called.
    _lazy_providers: Dict[str, str] = {}

    @staticmethod
    def register_lazy(name: str, dotted_path: str) -> None:
        """Register a provider without importing its module.

        Args:
            name: Provider name (e.g. 'gemini').
            dotted_path: ``"module.path:ClassName"`` resolved on first use.
        """
        ProviderFactory._lazy_providers[name.lower()] = dotted_path

    @staticmethod
    def _resolve(name: str) -> Type[ChatProvider]:
        """Resolve a provider name to its class, importing lazy entries on demand."""
        key = name.lower()
        if key in ProviderFactory._providers:
            return ProviderFactory._providers[key]
        lazy = ProviderFactory._lazy_providers.get(key)
        if lazy:
            module_path, attr = lazy.split(":", 1)
            import importlib
            cls = getattr(importlib.import_module(module_path), attr)
            ProviderFactory._providers[key] = cls  # cache for next time
            ProviderFactory._lazy_providers.pop(key, None)
            return cls
        raise ValueError(f"Provider '{name}' not registered")

    @staticmethod
    def register_provider(name: str, provider_class: Type[ChatProvider]) -> None:
        """
        Register a provider.

        Args:
            name (str): The name of the provider.
            provider_class (Type[ChatProvider]): The provider class.
        """
        ProviderFactory._providers[name] = provider_class

    @staticmethod
    def get_provider(name: str, api_key: Optional[str] = None, **kwargs) -> ChatProvider:
        """
        Get a provider instance.

        Args:
            name (str): The name of the provider (e.g., 'openai', 'ollama').
            api_key (Optional[str]): The API key for authentication.
                If None, providers will attempt to retrieve it via credgoo.
            **kwargs: Additional provider-specific arguments (e.g., base_url for Ollama).

        Returns:
            ChatProvider: The provider instance.

        Raises:
            ValueError: If the provider is not registered or initialization fails.
        """
        provider_name_lower = name.lower()  # Ensure consistent casing for lookup
        provider_class = ProviderFactory._resolve(provider_name_lower)
        try:
            # Pass the potentially retrieved api_key and other kwargs
            # Ensure api_key is only passed if it's expected by the constructor or not None
            # Most provider classes should accept api_key=None gracefully if not needed
            return provider_class(api_key=api_key, **kwargs)
        except TypeError as e:
            # Catch TypeError if api_key is passed unexpectedly or required but None
            # This might indicate an issue with the specific provider's __init__ signature
            raise ValueError(
                f"Failed to initialize provider '{name}' due to argument mismatch: {str(e)}") from e
        except ValueError:
            raise  # Re-raise explicit value errors from provider init
        except Exception as e:
            # Catch other potential initialization errors
            raise ValueError(
                f"Failed to initialize provider '{name}': {str(e)}") from e

    @staticmethod
    def list_providers() -> list:
        """
        List all registered providers.

        Returns:
            list: A list of provider names.
        """
        return list(set(ProviderFactory._providers) | set(ProviderFactory._lazy_providers))

    @staticmethod
    def list_models() -> Dict[str, list]:
        """
        List available models for all providers.

        Returns:
            Dict[str, list]: A dictionary mapping provider names to their available models.
            Returns an empty list for providers that don't implement list_models.
        """
        result = {}
        for name, provider_class in ProviderFactory._providers.items():
            try:
                result[name] = provider_class.list_models()
            except AttributeError:
                result[name] = []
        return result

    @staticmethod
    def get_provider_class(name: str) -> Type[ChatProvider]:
        """
        Get the provider class for a given provider name.

        Args:
            name (str): The name of the provider.

        Returns:
            Type[ChatProvider]: The provider class.

        Raises:
            ValueError: If the provider is not registered.
        """
        return ProviderFactory._resolve(name)
