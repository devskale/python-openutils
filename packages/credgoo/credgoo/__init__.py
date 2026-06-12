"""
credentials from google

A tool that gets credentials from Google Sheets via OAuth
"""

from importlib.metadata import version

from .credgoo import (
    get_api_key,
    setup_backend,
    oauth_flow,
    copy_template_sheet,
    get_api_key_from_sheets,
    cache_api_key,
)

__version__ = version("credgoo")
