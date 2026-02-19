import logging
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .uniioai import get_provider_api_key
from .errors import AuthenticationError

logger = logging.getLogger(__name__)

# Initialize the security scheme
security = HTTPBearer(auto_error=False)

def validate_proxy_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """
    Validates the bearer token provided to the proxy.
    
    This function currently acts as a pass-through for the token, which can be
    either a direct provider API key or a credgoo combined token (bearer@encryption).
    
    Future versions can implement centralized proxy-level authentication (e.g.,
    checking against a database of allowed users/tokens).
    
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
    
    Args:
        token: The bearer token (direct or credgoo combo).
        provider_name: The name of the LLM provider.
        
    Returns:
        str: The actual provider API key.
        
    Raises:
        HTTPException: 401 if key retrieval fails.
    """
    try:
        api_key = get_provider_api_key(token, provider_name)
        if not api_key and provider_name != 'ollama':
            raise AuthenticationError(f"No API key found for provider '{provider_name}'")
        return api_key
    except (ValueError, AuthenticationError) as e:
        logger.error(f"Authentication failed for {provider_name}: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        logger.exception(f"Unexpected error during authentication for {provider_name}")
        raise HTTPException(status_code=500, detail="Internal authentication error")
