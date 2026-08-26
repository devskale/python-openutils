import logging
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .provider_access import get_provider_api_key
from .config.instances import instance_requires_api_key
from .errors import AuthenticationError

logger = logging.getLogger(__name__)

# Initialize the security scheme
security = HTTPBearer(auto_error=False)

def validate_proxy_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Validate the bearer token provided to the proxy.

    Pass-through: the token must simply be PRESENT. It can be either a credgoo
    combined token ('bearer@encryption') or — historically — a direct provider
    key. Bare direct keys are rejected in verify_provider_access (see there);
    this layer only guards against missing credentials.

    Args:
        credentials: The HTTPBearer credentials from the request.

    Returns:
        str: The validated token string.

    Raises:
        HTTPException: 401 if authentication is missing.
    """
    if not credentials or not credentials.credentials:
        logger.warning("Authentication missing in request")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a Bearer token (provider key or credgoo combo)."
        )
    return credentials.credentials

def get_optional_proxy_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """
    Optionally returns the bearer token if provided, without raising an error.
    
    Args:
        credentials: The HTTPBearer credentials from the request.
        
    Returns:
        Optional[str]: The token string if present, otherwise None.
    """
    if credentials and credentials.credentials:
        return credentials.credentials
    return None

def verify_provider_access(token: str, provider_name: str) -> str:
    """
    Verifies that the provided token can be used to retrieve an API key for the provider.

    Hard gate (live finding 2026-08-26): bare (non-'@') tokens are rejected for
    keyed providers. A bare string used to slip through as a "direct key" — but
    keyed providers like TU are served from the process-wide POOLED client whose
    Authorization header carries the SERVER's own credentials, so any junk
    bearer got served on our quota once the pool was warm. Consumers must
    present a credgoo combined token ('bearer@encryption'), which resolves
    server-side via credgoo.

    Args:
        token: The bearer token (direct or credgoo combo).
        provider_name: The name of the LLM provider.

    Returns:
        str: The actual provider API key.

    Raises:
        HTTPException: 401 if key retrieval fails.
    """
    try:
        if token and "@" not in token and instance_requires_api_key(provider_name):
            logger.warning("Rejected bare (non-combo) bearer token for '%s'", provider_name)
            raise AuthenticationError(
                "Direct provider keys are not accepted on this gateway; "
                "send a credgoo combined token (bearer@encryption)."
            )
        api_key = get_provider_api_key(token, provider_name)
        if not api_key and instance_requires_api_key(provider_name):
            raise AuthenticationError(f"No API key found for provider '{provider_name}'")
        return api_key
    except (ValueError, AuthenticationError) as e:
        logger.error(f"Authentication failed for {provider_name}: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        logger.exception(f"Unexpected error during authentication for {provider_name}")
        raise HTTPException(status_code=500, detail="Internal authentication error")
