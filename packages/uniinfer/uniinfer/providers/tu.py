"""
OpenAIcompliant TU provider implementation.
"""
import json
import requests
from requests.exceptions import Timeout, ConnectionError
import time
import threading
import logging
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error, ProviderError

logger = logging.getLogger(__name__)


class TuAIProvider(ChatProvider):
    """
    Provider for OpenAI API.
    """

    BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    # Throttling state
    _last_request_end_time = 0.0
    _lock = threading.Lock()
    _min_gap = 1.5  # 1.5 seconds gap between end of one call and start of next

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            organization (Optional[str]): The OpenAI organization ID.
        """
        super().__init__(api_key)
        self.organization = organization

    @classmethod
    def _throttle(cls):
        """Enforce rate limiting: wait until (last_end_time + min_gap) <= current_time."""
        with cls._lock:
            current_time = time.time()
            # Calculate when we can safely start the next request
            next_allowed_start = cls._last_request_end_time + cls._min_gap

            if current_time < next_allowed_start:
                sleep_time = next_allowed_start - current_time
                logger.debug(
                    f"Throttling request. Sleeping for {sleep_time:.4f}s")
                time.sleep(sleep_time)

            # Note: We update _last_request_end_time AFTER the request finishes,
            # but here we don't know when it finishes.
            # To be safe and simple, we can update it to current_time (start of this request)
            # but the requirement is "gap between end of message and next message".
            # So we need to update _last_request_end_time at the END of complete() and stream_complete().

    @classmethod
    def _mark_request_end(cls):
        """Mark the end time of a request."""
        with cls._lock:
            cls._last_request_end_time = time.time()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from OpenAI using the API.

        Returns:
            list: A list of available model IDs.
        """
        cls._throttle()

        if not api_key:
            raise ValueError("API key is required to list models")
        url = f"{cls.BASE_URL}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise map_provider_error("TU", Exception(
                f"OpenAI API error: {response.status_code} - {response.text}"), status_code=response.status_code, response_body=response.text)

        data = response.json()
        return [model["id"] for model in data.get("data", [])]

    def _flatten_messages(self, messages: list) -> list:
        """
        Process messages. Flatten text-only lists if needed, but preserve image/multimodal content.
        """
        processed_messages = []
        for msg in messages:
            msg_dict = msg.to_dict()
            content = msg_dict.get("content")

            if isinstance(content, list):
                # Check if it contains any non-text types (like image_url)
                has_non_text = any(
                    isinstance(part, dict) and part.get("type") != "text"
                    for part in content
                )

                if has_non_text:
                    # Preserve structure for VLMs (Vision Language Models)
                    pass
                else:
                    # Flatten text-only lists
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))

                    # If we found text parts, join them
                    if text_parts:
                        msg_dict["content"] = "".join(text_parts)

            processed_messages.append(msg_dict)
        return processed_messages

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
        self._throttle()

        try:
            logger.info(
                f"Starting complete request for model: {request.model}")
            if self.api_key is None:
                raise ValueError("OpenAI API key is required")

            endpoint = f"{self.BASE_URL}/chat/completions"

            # Prepare the request payload
            payload = {
                # Default model if none specified
                "model": request.model or "openai/RedHatAI/DeepSeek-R1-0528-quantized.w4a16",
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
                headers["TUW-Organization"] = self.organization

            try:
                logger.debug(f"Sending POST request to {endpoint}")
                start_time = time.time()
                response = requests.post(
                    endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=300  # Increase timeout to 300s
                )
                duration = time.time() - start_time
                logger.info(
                    f"Request finished in {duration:.2f}s with status {response.status_code}")
            except Timeout as e:
                logger.error(f"Request timed out after 300s: {e}")
                raise map_provider_error("TU", e, status_code=408)
            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                raise map_provider_error("TU", e, status_code=503)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception: {e}")
                raise map_provider_error("TU", e, status_code=500)

            # Handle error response
            if response.status_code != 200:
                error_msg = f"TU API error: {response.status_code} - {response.text}"
                raise map_provider_error("TU", Exception(
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
                provider='tu',
                model=response_data.get('model', request.model),
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )
        finally:
            self._mark_request_end()

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
        self._throttle()

        try:
            logger.info(
                f"Starting stream_complete request for model: {request.model}")
            if self.api_key is None:
                raise ValueError("OpenAI API key is required")

            endpoint = f"{self.BASE_URL}/chat/completions"

            # Prepare the request payload
            payload = {
                "model": request.model or "openai/RedHatAI/DeepSeek-R1-0528-quantized.w4a16",
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
                headers["TUW-Organization"] = self.organization

            try:
                logger.debug(f"Sending streaming POST request to {endpoint}")
                start_time = time.time()
                with requests.post(
                    endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    stream=True,
                    timeout=300  # Increase timeout to 300s
                ) as response:
                    duration = time.time() - start_time
                    logger.info(
                        f"Stream request connection established in {duration:.2f}s with status {response.status_code}")

                    # Handle error response
                    if response.status_code != 200:
                        error_text = response.text
                        if "504 Gateway Time-out" in error_text:
                            error_msg = f"TU API error: {response.status_code} - Gateway Timeout (Model may be overloaded)"
                        else:
                            error_msg = f"TU API error: {response.status_code} - {error_text}"
                        raise map_provider_error("TU", Exception(
                            error_msg), status_code=response.status_code, response_body=error_text)

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

                                    # Skip if neither content nor tool_calls present
                                    if 'delta' not in choice:
                                        continue
                                    delta = choice['delta']
                                    if not delta.get('content') and not delta.get('tool_calls'):
                                        continue

                                    # Get content and tool_calls from delta
                                    content = choice['delta'].get('content')
                                    tool_calls = choice['delta'].get(
                                        'tool_calls')

                                    # Create a message for this chunk
                                    message = ChatMessage(
                                        role=choice['delta'].get(
                                            'role', 'assistant'),
                                        content=content,
                                        tool_calls=tool_calls
                                    )

                                    # Usage stats typically not provided in stream chunks
                                    usage = {}

                                    yield ChatCompletionResponse(
                                        message=message,
                                        provider='tu',
                                        model=data.get('model', request.model),
                                        usage=usage,
                                        raw_response=data
                                    )
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                            except Exception as e:
                                # Skip other errors in individual chunks
                                continue
            except Timeout as e:
                logger.error(f"Stream request timed out after 300s: {e}")
                raise map_provider_error("TU", e, status_code=408)
            except ConnectionError as e:
                logger.error(f"Stream connection error: {e}")
                raise map_provider_error("TU", e, status_code=503)
            except requests.exceptions.RequestException as e:
                logger.error(f"Stream request exception: {e}")
                raise map_provider_error("TU", e, status_code=500)
        finally:
            self._mark_request_end()
