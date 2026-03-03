"""Compatibility shim for provider configs.

Use `uniinfer.config.providers` for runtime imports.
This module remains for backward compatibility with older example imports.
"""

from uniinfer.config.providers import (  # noqa: F401
    PROVIDER_CONFIGS,
    add_provider,
    get_all_providers,
    get_provider_config,
)
