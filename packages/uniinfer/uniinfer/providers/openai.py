"""
OpenAI provider implementation.
"""
import json
import requests
from typing import Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class OpenAIProvider(ChatProvider):
    """
    Provider for OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            organization (Optional[str]): The OpenAI organization ID.
        """
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.organization = organization

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenAI using the API.

        Returns:
            list: A list of available model IDs.
        """
        if not api_key:
            raise ValueError("API key is required to list models")

        url = "https://api.openai.com/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise map_provider_error("OpenAI", Exception(
                    f"OpenAI API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(
                e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(
                e, 'response') else str(e)
            raise map_provider_error(
                "OpenAI", e, status_code=status_code, response_body=response_body)

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
        Make a chat completion request to OpenAI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional OpenAI-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model or "gpt-3.5-turbo",  # Default model if none specified
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

        # Add any provider-specific parameters (like functions, tools, etc.)
        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Add organization header if provided
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload)
            )

            # Handle error response
            if response.status_code != 200:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                raise map_provider_error("OpenAI", Exception(
                    error_msg), status_code=response.status_code, response_body=response.text)

            # Parse the response
            response_data = response.json()
            choice = response_data['choices'][0]
            message = ChatMessage(
                role=choice['message']['role'],
                content=choice['message'].get('content'),
                tool_calls=choice['message'].get('tool_calls')
            )

            return ChatCompletionResponse(
                message=message,
                provider='openai',
                model=response_data.get('model', request.model),
                usage=response_data.get('usage', {}),
                raw_response=response_data,
                finish_reason=choice.get('finish_reason')
            )
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(
                e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(
                e, 'response') else str(e)
            raise map_provider_error(
                "OpenAI", e, status_code=status_code, response_body=response_body)

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from OpenAI.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional OpenAI-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("OpenAI API key is required")

        endpoint = f"{self.base_url}/chat/completions"

        # Prepare the request payload
        payload = {
            "model": request.model or "gpt-3.5-turbo",
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

        # Add organization header if provided
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        try:
            with requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                stream=True
            ) as response:
                # Handle error response
                if response.status_code != 200:
                    error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                    raise map_provider_error("OpenAI", Exception(
                        error_msg), status_code=response.status_code, response_body=response.text)

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        # Parse the JSON data from the stream
                        try:
                            line = line.decode('utf-8')

                            # Skip empty lines, data: [DONE], or invalid lines
                            if not line or line == 'data: [DONE]' or not line.startswith('data: '):
                                continue

                            # Parse the data portion
                            data_str = line[6:]  # Remove 'data: ' prefix
                            data = json.loads(data_str)

                            if len(data['choices']) > 0:
                                choice = data['choices'][0]
                                finish_reason = choice.get('finish_reason')
                                delta = choice.get('delta', {})

                                # Skip if neither content nor tool_calls present AND no finish_reason
                                if not delta.get('content') and not delta.get('tool_calls') and not finish_reason:
                                    continue

                                # Get content and tool_calls from delta
                                content = delta.get('content')
                                tool_calls = delta.get('tool_calls')

                                # Create a message for this chunk
                                message = ChatMessage(
                                    role=delta.get('role', 'assistant'),
                                    content=content,
                                    tool_calls=tool_calls
                                )

                                # Usage stats typically not provided in stream chunks
                                usage = {}

                                yield ChatCompletionResponse(
                                    message=message,
                                    provider='openai',
                                    model=data.get('model', request.model),
                                    usage=usage,
                                    raw_response=data,
                                    finish_reason=finish_reason
                                )
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                        except Exception:
                            # Skip other errors in individual chunks
                            continue
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(
                e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(
                e, 'response') else str(e)
            raise map_provider_error(
                "OpenAI", e, status_code=status_code, response_body=response_body)
