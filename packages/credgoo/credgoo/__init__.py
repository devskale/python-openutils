"""
credentials from goo

A tool that gets credentials from pluggable backends (gdrive, airtable, ...).
"""

from importlib.metadata import version

from .credgoo import (
    get_api_key,
    setup_backend,
    store_credentials,
    load_credentials,
    cache_api_key,
)

__version__ = version("credgoo")
