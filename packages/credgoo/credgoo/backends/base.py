"""Abstract base for credgoo backends."""

from abc import ABC, abstractmethod
from pathlib import Path


class CredgooBackend(ABC):
    """Each backend must implement setup() and fetch_key()."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier stored in credgoo.txt as 'backend' field."""

    @abstractmethod
    def setup(self, cache_dir: Path) -> bool:
        """Interactive first-time setup. Returns True on success.
        Responsible for writing credgoo.txt via store_credentials()."""

    @abstractmethod
    def fetch_key(self, service: str, creds: dict) -> str | None:
        """Retrieve a single API key for *service*.
        *creds* is the full dict loaded from credgoo.txt.
        Returns plaintext key or None."""
