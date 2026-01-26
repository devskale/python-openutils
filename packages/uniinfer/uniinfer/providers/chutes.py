"""
Chutes provider implementation.

Chutes is a unified API to access multiple AI models from different providers.
"""
import json
import requests
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class ChutesProvider(ChatProvider):
    """
    Provider for Chutes API.

    Chutes provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Chutes provider.

        Args:
            api_key (Optional[str]): The Chutes API key.
        """
        super().__init__(api_key)
        self.base_url = "https://llm.chutes.ai/v1"  # Updated to match reference

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Chutes.

        Args:
            api_key (Optional[str]): The Chutes API key. If not provided,
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
                api_key = get_api_key('chutes')
            except ImportError:
                raise ValueError(
                    "credgoo not installed. Please provide an API key or install credgoo.")
            except Exception as e:
                raise ValueError(
                    f"Failed to get Chutes API key from credgoo: {e}")

        if not api_key:
            raise ValueError(
                "Chutes API key is required. Provide it directly or configure credgoo.")

        endpoint = "https://llm.chutes.ai/v1/models"  # Updated to match reference

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.get(endpoint, headers=headers)

        if response.status_code != 200:
            error_msg = f"Chutes API error: {response.status_code} - {response.text}"
            raise map_provider_error("Chutes", Exception(error_msg), status_code=response.status_code, response_body=response.text)

        models = response.json().get('data', [])
        
        # Only return the model IDs
        # Only return the model IDs
        return [model['id'] for model in models]

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
        Make a chat completion request to Chutes.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Chutes-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Chutes API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            # Default model if none specified
            "model": request.model or "deepseek-ai/DeepSeek-V3-0324",
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
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload)
        )

        # Handle error response
        if response.status_code != 200:
            error_msg = f"Chutes API error: {response.status_code} - {response.text}"
            raise map_provider_error("Chutes", Exception(error_msg), status_code=response.status_code, response_body=response.text)

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
            provider='chutes',
            model=actual_model,
            usage=response_data.get('usage', {}),
            raw_response=response_data
        )

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Chutes.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Chutes-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Chutes API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model or "deepseek-ai/DeepSeek-V3-0324",
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
            "Authorization": f"Bearer {self.api_key}"
        }

        with requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        ) as response:
            if response.status_code != 200:
                error_msg = f"Chutes API error: {response.status_code} - {response.text}"
                raise map_provider_error("Chutes", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8').strip()
                        if not line or not line.startswith('data: '):
                            continue
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        chunk = data.strip()
                        if not chunk:
                            continue
                        data_json = json.loads(chunk)
                        if 'choices' not in data_json or not data_json['choices']:
                            continue
                        choice = data_json['choices'][0]
                        
                        if 'delta' not in choice:
                            continue
                        
                        delta = choice['delta']
                        
                        # Skip if neither content nor tool_calls present
                        if not delta.get('content') and not delta.get('tool_calls'):
                            continue

                        # Extract content delta and tool_calls
                        content = delta.get('content')
                        tool_calls = delta.get('tool_calls')
                        role = delta.get('role', 'assistant')
                        
                        message = ChatMessage(
                            role=role,
                            content=content,
                            tool_calls=tool_calls
                        )
                        usage = {}
                        model = data_json.get('model', request.model)
                        yield ChatCompletionResponse(
                            message=message,
                            provider='chutes',
                            model=model,
                            usage=usage,
                            raw_response=data_json
                        )
                    except json.JSONDecodeError:
                        continue
                        content = choice['delta']['content']

                        # Get role from delta or default to assistant
                        role = choice['delta'].get('role', 'assistant')

                        # Create a message for this chunk
                        message = ChatMessage(role=role, content=content)

                        # Usage stats typically not provided in stream chunks
                        usage = {}

                        # Get model info if available
                        model = data.get('model', request.model)

                        yield ChatCompletionResponse(
                            message=message,
                            provider='chutes',
                            model=model,
                            usage=usage,
                            raw_response=data
                        )
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
