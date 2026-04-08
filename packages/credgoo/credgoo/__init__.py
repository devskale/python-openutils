"""
credentials from google

A tool that gets credentials from google sheets
"""

from importlib.metadata import version

from .credgoo import (
    decrypt_key,
    get_api_key_from_google,
    cache_api_key,
    get_api_key
)

__version__ = version("credgoo")
