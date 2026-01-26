"""
Error handling for UniInfer.
"""
from typing import Optional

class UniInferError(Exception):
    """Base exception for all UniInfer errors."""
    pass


class ProviderError(UniInferError):
    """Error related to a provider operation."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(ProviderError):
    """Authentication error with a provider."""
    pass


class RateLimitError(ProviderError):
    """Rate limit error from a provider."""
    pass


class TimeoutError(ProviderError):
    """Timeout error from a provider."""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request error."""
    pass


def map_provider_error(provider_name: str, original_error: Exception, status_code: Optional[int] = None, response_body: Optional[str] = None) -> ProviderError:
    """
    Map a provider-specific error to a UniInfer error.
    
    Args:
        provider_name (str): The name of the provider.
        original_error (Exception): The original error.
        status_code (Optional[int]): The HTTP status code from the provider.
        response_body (Optional[str]): The raw response body from the provider.
        
    Returns:
        ProviderError: A standardized UniInfer error.
    """
    error_message = str(original_error).lower()
    
    # Common authentication errors
    if status_code == 401 or any(term in error_message for term in ["authentication", "auth", "unauthorized", "api key", "401"]):
        return AuthenticationError(f"{provider_name} authentication error: {str(original_error)}", status_code, response_body)
    
    # Rate limit errors
    if status_code == 429 or any(term in error_message for term in ["rate limit", "ratelimit", "too many requests", "429"]):
        return RateLimitError(f"{provider_name} rate limit error: {str(original_error)}", status_code, response_body)
    
    # Timeout errors
    if status_code in [408, 504] or any(term in error_message for term in ["timeout", "timed out"]):
        return TimeoutError(f"{provider_name} timeout error: {str(original_error)}", status_code, response_body)
    
    # Invalid request errors
    if status_code == 400 or any(term in error_message for term in ["invalid", "validation", "bad request", "400"]):
        return InvalidRequestError(f"{provider_name} invalid request: {str(original_error)}", status_code, response_body)
    
    # Default to generic provider error
    return ProviderError(f"{provider_name} error: {str(original_error)}", status_code, response_body)
