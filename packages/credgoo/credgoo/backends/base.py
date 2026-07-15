"""Abstract base for credgoo backends.

Declares the full backend contract. ``setup`` and ``fetch_key`` are mandatory.
The key-management operations (add/delete/clear/list/dedupe) are optional
capabilities: a backend advertises support for one simply by overriding it.
The default implementation raises :class:`UnsupportedOperation`, and
:meth:`CredgooBackend.supports` reports — side-effect free — whether a backend
overrides a given capability.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class UnsupportedOperation(Exception):
    """Raised when a backend is asked for a capability it does not provide.

    Carries the backend name and capability so callers can report it cleanly.
    """

    def __init__(self, backend_name, capability):
        self.backend_name = backend_name
        self.capability = capability
        super().__init__(f"backend '{backend_name}' does not support '{capability}'")


class CredgooBackend(ABC):
    """A source of API keys.

    Mandatory: :meth:`setup` and :meth:`fetch_key`. Optional: the
    key-management methods, which default to unsupported — override the ones
    your backend provides. Use :meth:`supports` to ask whether a capability is
    available before calling it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier stored in credgoo.txt as the backend field."""

    @abstractmethod
    def setup(self, cache_dir: Path) -> bool:
        """Interactive first-time setup. Returns True on success.
        Responsible for writing credgoo.txt via the injected _store_creds helper."""

    @abstractmethod
    def fetch_key(self, service: str, creds: dict) -> str | None:
        """Retrieve a single API key for *service*.
        *creds* is this backend's dict from credgoo.txt.
        Returns the plaintext key or None."""

    @property
    def cache_integrity(self) -> bool:
        """Whether cached entries for this backend should carry an integrity tag
        (HMAC) so staleness — e.g. credential rotation — is detected and the
        cache invalidated on read. Default off; override to enable."""
        return False

    def cache_key(self, creds: dict) -> str | None:
        """Derive the local-cache encryption key from *creds*.
        Default: the stored 'encryption_key' field. Override for a different
        scheme (e.g. deriving from a token)."""
        return creds.get("encryption_key")

    def add_key(self, service: str, api_key: str, creds: dict) -> bool:
        """Add or update a key. Override to support; default raises unsupported."""
        raise UnsupportedOperation(self.name, "add_key")

    def delete_key(self, service: str, creds: dict) -> bool:
        """Delete a key. Override to support; default raises unsupported."""
        raise UnsupportedOperation(self.name, "delete_key")

    def clear_key(self, service: str, creds: dict) -> bool:
        """Blank a key. Override to support; default raises unsupported."""
        raise UnsupportedOperation(self.name, "clear_key")

    def list_keys(self, creds: dict) -> list:
        """Return all service names. Override to support; default raises unsupported."""
        raise UnsupportedOperation(self.name, "list_keys")

    def dedupe_keys(self, creds: dict) -> tuple:
        """Remove duplicate services; return (kept, removed). Default raises unsupported."""
        raise UnsupportedOperation(self.name, "dedupe_keys")

    def supports(self, capability: str) -> bool:
        """True if this backend overrides the default (unsupported) *capability*.

        The contract guarantees every capability names a method on the type, so
        this is an override check against the base — not a probe for whether the
        method exists.
        """
        return getattr(type(self), capability) is not getattr(CredgooBackend, capability)
