"""
OpenRouter provider implementation.

OpenRouter is a unified API to access multiple AI models from different providers.
"""
import json
import requests
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class OpenRouterProvider(ChatProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key (Optional[str]): The OpenRouter API key.
        """
        super().__init__(api_key)
        self.base_url = "https://openrouter.ai/api/v1"

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenRouter.

        Args:
            api_key (Optional[str]): The OpenRouter API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available free model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        # Determine the API key to use
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key('openrouter')
            except ImportError:
                raise ValueError(
                    "credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(
                    f"Failed to get OpenRouter API key from credgoo: {e}")

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://openrouter.ai/api/v1/models"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer"
        }

        try:
            response = requests.get(endpoint, headers=headers)

            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            models = response.json().get('data', [])
            free_models = [
                model['id']
                for model in models
                if model.get('pricing', {}).get('prompt') == '0'
                and model.get('pricing', {}).get('completion') == '0'
            ]
            return free_models
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)

    def _flatten_messages(self, messages: list) -> list:
        """
        Flatten message content if it's a list of text objects.
        Some proxies/providers only support string content.
        """
        flattened_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")
            
            if isinstance(content, list):
                # Check if it's a list of text objects
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                
                # If we found text parts, join them
                if text_parts:
                    msg_dict["content"] = "".join(text_parts)
            
            flattened_messages.append(msg_dict)
        return flattened_messages

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to OpenRouter.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional OpenRouter-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("OpenRouter API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            # Default model if none specified
            "model": request.model or "moonshotai/moonlight-16b-a3b-instruct:free",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",  # Required by OpenRouter
            "X-Title": "UniInfer"  # Application name for OpenRouter
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload)
            )

            # Handle error response
            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            # Parse the response
            response_data = response.json()
            choice = response_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )

            # Get the actual model used from the response
            actual_model = response_data.get('model', request.model)

            return ChatCompletionResponse(
                message=message,
                provider='openrouter',
                model=actual_model,
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from OpenRouter.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional OpenRouter-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("OpenRouter API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model or "moonshotai/moonlight-16b-a3b-instruct:free",
            "messages": self._flatten_messages(request.messages),
            "temperature": request.temperature,
            "stream": True
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/uniinfer",
            "X-Title": "UniInfer"
        }

        try:
            with requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                stream=True
            ) as response:
                # Handle error response
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                    raise map_provider_error("OpenRouter", Exception(error_msg), status_code=response.status_code, response_body=response.text)

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        # Parse the JSON data from the stream
                        try:
                            line = line.decode('utf-8')

                            # Skip empty lines or [DONE]
                            if not line or line == 'data: [DONE]':
                                continue

                            # Remove 'data: ' prefix if present
                            if line.startswith('data: '):
                                line = line[6:]

                            data = json.loads(line)

                            # Skip non-content deltas
                            if 'choices' not in data or not data['choices']:
                                continue

                            choice = data['choices'][0]

                            if 'delta' not in choice:
                                continue
                            
                            delta = choice['delta']
                            
                            # Skip if neither content nor tool_calls present
                            if not delta.get('content') and not delta.get('tool_calls'):
                                continue

                            # Extract content delta and tool_calls
                            content = delta.get('content')
                            tool_calls = delta.get('tool_calls')

                            # Get role from delta or default to assistant
                            role = choice['delta'].get('role', 'assistant')

                            # Create a message for this chunk
                            message = ChatMessage(
                                role=role,
                                content=content,
                                tool_calls=tool_calls
                            )

                            # Usage stats typically not provided in stream chunks
                            usage = {}

                            # Get model info if available
                            model = data.get('model', request.model)

                            yield ChatCompletionResponse(
                                message=message,
                                provider='openrouter',
                                model=model,
                                usage=usage,
                                raw_response=data
                            )
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("OpenRouter", e, status_code=status_code, response_body=response_body)
