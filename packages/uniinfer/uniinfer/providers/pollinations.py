"""
Pollinations provider implementation.

Pollinations is a unified API to access multiple AI models from different providers.
"""
from typing import Optional, Iterator
import requests
import json
import urllib.parse

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error


class PollinationsProvider(ChatProvider):
    """
    Provider for Pollinations API.

    Pollinations provides a unified interface to access multiple AI models from
    different providers, including Anthropic, OpenAI, and more.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Pollinations provider.

        Args:
            api_key (Optional[str]): The Pollinations API key.
        """
        super().__init__(api_key)
        # Use new API if proper API key is provided, otherwise use legacy API
        if api_key and (api_key.startswith('plln_pk_') or api_key.startswith('plln_sk_')):
            self.base_url = "https://enter.pollinations.ai/api/generate/v1/chat/completions"
            self.use_new_api = True
        else:
            self.base_url = "https://text.pollinations.ai"
            self.use_new_api = False

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Pollinations.

        Args:
            api_key (Optional[str]): The Pollinations API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        # Try new API first if proper key is provided
        use_new_api = False
        if api_key and (api_key.startswith('plln_pk_') or api_key.startswith('plln_sk_')):
            use_new_api = True
        else:
            try:
                from credgoo.credgoo import get_api_key
                key = get_api_key('pollinations')
                if key and (key.startswith('plln_pk_') or key.startswith('plln_sk_')):
                    api_key = key
                    use_new_api = True
            except (ImportError, Exception):
                pass

        if use_new_api:
            endpoint = "https://enter.pollinations.ai/api/generate/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(endpoint, headers=headers)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            models = response.json()
            if isinstance(models, dict) and 'data' in models:
                return [model.get('id', '') for model in models['data'] if model.get('id')]
        else:
            # Use legacy API for anonymous users
            endpoint = "https://text.pollinations.ai/models"
            response = requests.get(endpoint)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            models = response.json()
            if isinstance(models, list) and len(models) > 0 and isinstance(models[0], dict):
                return [model.get('name', '') for model in models if model.get('name')]

        return []

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a text generation request to Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.use_new_api:
            return self._complete_new_api(request, **provider_specific_kwargs)
        else:
            return self._complete_legacy_api(request, **provider_specific_kwargs)

    def _complete_new_api(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Complete using new OpenAI-compatible API."""
        messages = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            
            messages.append({
                "role": msg.role,
                "content": content
            })

        payload = {
            "model": request.model or "openai",
            "messages": messages,
            "stream": False
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            response_data = response.json()

            if 'choices' not in response_data or not response_data['choices']:
                raise Exception("API returned invalid response")

            choice = response_data['choices'][0]
            content = choice.get('message', {}).get('content', '')
            
            message = ChatMessage(
                role="assistant",
                content=content
            )

            return ChatCompletionResponse(
                message=message,
                provider='pollinations',
                model=request.model,
                usage=response_data.get('usage', {}),
                raw_response=response_data
            )

        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("Pollinations", e, status_code=status_code, response_body=response_body)

    def _complete_legacy_api(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Complete using legacy text API."""
        import urllib.parse
        
        def _flatten_text(content):
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return "".join(parts)
            return content

        last_user_message = None
        system_message = None
        for msg in request.messages:
            if msg.role == "user" and not last_user_message:
                last_user_message = _flatten_text(msg.content)
            elif msg.role == "system" and not system_message:
                system_message = _flatten_text(msg.content)
        
        if not last_user_message:
            raise ValueError("At least one user message is required for Pollinations API.")

        params = {
            "model": request.model or "openai",
            "seed": 42,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        if system_message:
            params["system"] = system_message

        params.update(provider_specific_kwargs)

        encoded_prompt = urllib.parse.quote(last_user_message)
        url = f"{self.base_url}/{encoded_prompt}"

        try:
            response = requests.get(url, params=params)

            if response.status_code != 200:
                error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

            if params.get("json") == "true":
                try:
                    response_data = json.loads(response.text)
                except json.JSONDecodeError:
                    raise Exception("API returned invalid JSON string")
            else:
                response_data = {"result": response.text}

            message = ChatMessage(
                role="assistant",
                content=response_data.get("result", "")
            )

            return ChatCompletionResponse(
                message=message,
                provider='pollinations',
                model=request.model,
                usage={},
                raw_response=response_data
            )

        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("Pollinations", e, status_code=status_code, response_body=response_body)

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Pollinations.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Pollinations-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.use_new_api:
            yield from self._stream_new_api(request, **provider_specific_kwargs)
        else:
            yield from self._stream_legacy_api(request, **provider_specific_kwargs)

    def _stream_new_api(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """Stream using new OpenAI-compatible API."""
        messages = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            
            messages.append({
                "role": msg.role,
                "content": content
            })

        payload = {
            "model": request.model or "openai",
            "messages": messages,
            "stream": True
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        payload.update(provider_specific_kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            with requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                stream=True
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                    raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

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
                            content = delta.get('content', '')
                            role = delta.get('role', 'assistant')
                            
                            if not content:
                                continue
                            
                            message = ChatMessage(role=role, content=content)
                            usage = {}
                            model = data_json.get('model', request.model)
                            yield ChatCompletionResponse(
                                message=message,
                                provider='pollinations',
                                model=model,
                                usage=usage,
                                raw_response=data_json
                            )
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("Pollinations", e, status_code=status_code, response_body=response_body)

    def _stream_legacy_api(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """Stream using legacy text API."""
        import urllib.parse
        
        def _flatten_text(content):
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return "".join(parts)
            return content

        last_user_message = None
        system_message = None
        for msg in request.messages:
            if msg.role == "user" and not last_user_message:
                last_user_message = _flatten_text(msg.content)
            elif msg.role == "system" and not system_message:
                system_message = _flatten_text(msg.content)
        
        if not last_user_message:
            raise ValueError("At least one user message is required for Pollinations API.")

        params = {
            "model": request.model or "openai",
            "seed": 42,
            "stream": True
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        if system_message:
            params["system"] = system_message

        params.update(provider_specific_kwargs)

        encoded_prompt = urllib.parse.quote(last_user_message)
        url = f"{self.base_url}/{encoded_prompt}"

        try:
            with requests.get(url, params=params, stream=True) as response:
                if response.status_code != 200:
                    error_msg = f"Pollinations API error: {response.status_code} - {response.text}"
                    raise map_provider_error("Pollinations", Exception(error_msg), status_code=response.status_code, response_body=response.text)

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
                            if 'delta' not in choice or 'content' not in choice['delta'] or not choice['delta']['content']:
                                continue
                            content = choice['delta']['content']
                            role = choice['delta'].get('role', 'assistant')
                            message = ChatMessage(role=role, content=content)
                            usage = {}
                            model = data_json.get('model', request.model)
                            yield ChatCompletionResponse(
                                message=message,
                                provider='pollinations',
                                model=model,
                                usage=usage,
                                raw_response=data_json
                            )
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("Pollinations", e, status_code=status_code, response_body=response_body)
