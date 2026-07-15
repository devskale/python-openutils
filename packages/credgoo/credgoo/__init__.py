"""
credentials from goo

A tool that gets credentials from pluggable backends (gdrive, airtable, ...).
"""

from importlib.metadata import version

from .credgoo import get_api_key, setup_backend
from .store import (
    CredentialStore,
    cache_api_key,
    load_credentials,
    store_credentials,
)

__version__ = version("credgoo")
