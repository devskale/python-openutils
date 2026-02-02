"""
HuggingFace Inference provider implementation.
"""
from typing import Dict, Any, Iterator, Optional, List, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import AuthenticationError, map_provider_error, UniInferError

try:
    from huggingface_hub import InferenceClient, AsyncInferenceClient, HfApi
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


class HuggingFaceProvider(ChatProvider):
    """
    Provider for HuggingFace Inference API.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the HuggingFace provider.

        Args:
            api_key (Optional[str]): The HuggingFace API key.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)

        if not HAS_HUGGINGFACE:
            raise ImportError(
                "huggingface_hub package is required for the HuggingFaceProvider. "
                "Install it with: pip install huggingface_hub"
            )

        self.client = InferenceClient(token=self.api_key)
        self.async_client = AsyncInferenceClient(token=self.api_key)
        self.hf_api = HfApi(token=self.api_key)

    async def aclose(self):
        """Close the HuggingFace async client."""
        # AsyncInferenceClient currently doesn't have an explicit close in all versions
        # but it inherits from a base that uses httpx. Check if it needs explicit closing.
        await super().aclose()

    def _get_pipeline_tag(self, model_id: str) -> Optional[str]:
        """Retrieve the pipeline tag for the given model."""
        try:
            info = self.hf_api.model_info(model_id)
            return getattr(info, "pipeline_tag", None)
        except Exception:
            return None

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> List[str]:
        """List available models from HuggingFace."""
        try:
            if not HAS_HUGGINGFACE:
                return []

            if not api_key:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("huggingface")

            hf_api = HfApi(token=api_key)
            models = hf_api.list_models(pipeline_tag="text-generation", sort="likes", limit=100)
            return [model.id for model in models if model.id]
        except Exception:
            return [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-70B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.1"
            ]

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to HuggingFace Inference API."""
        if self.api_key is None:
            raise ValueError("HuggingFace API key is required")

        if not request.model:
            raise ValueError("Model must be specified for HuggingFace Inference")

        model_id = request.model.split()[0].split(":", 1)[0]

        try:
            def _flatten(content: Any) -> str:
                if isinstance(content, list):
                    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                return str(content) if content is not None else ""

            flat_messages = []
            for m in request.messages:
                flat_messages.append({"role": m.role, "content": _flatten(m.content)})

            # Use chat_completion as primary method
            extra = {}
            if "SmolLM" in model_id:
                extra["xml_tools"] = request.tools
                extra["python_tools"] = request.tools

            try:
                resp = await self.async_client.chat_completion(
                    messages=flat_messages,
                    model=model_id,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature or 0.7,
                    **extra
                )
            except Exception as e:
                # Fallback to text_generation if conversational is not available
                if "Not Found" in str(e) or "conversational" in str(e):
                    last_user_msg = next((_flatten(m.content) for m in reversed(request.messages) if m.role == "user"), "")
                    completion = await self.async_client.text_generation(
                        prompt=last_user_msg,
                        model=model_id,
                        max_new_tokens=request.max_tokens or 1024,
                        temperature=request.temperature or 0.7,
                        **provider_specific_kwargs
                    )
                    message = ChatMessage(role="assistant", content=completion)
                    return ChatCompletionResponse(
                        message=message,
                        provider='huggingface',
                        model=model_id,
                        usage={},
                        raw_response={"content": completion}
                    )
                raise

            assistant_msg = resp.choices[0].message
            content = assistant_msg.content
            tool_calls = getattr(assistant_msg, "tool_calls", None)

            message = ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
            
            return ChatCompletionResponse(
                message=message,
                provider='huggingface',
                model=model_id,
                usage=getattr(resp, "usage", {}),
                raw_response=resp
            )

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', None)
                response_body = getattr(e.response, 'text', str(e))
            raise map_provider_error("huggingface", e, status_code=status_code, response_body=response_body)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from HuggingFace Inference API."""
        if self.api_key is None:
            raise ValueError("HuggingFace API key is required")

        model_id = request.model.split()[0].split(":", 1)[0]

        try:
            def _flatten(content: Any) -> str:
                if isinstance(content, list):
                    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                return str(content) if content is not None else ""

            flat_messages = []
            for m in request.messages:
                flat_messages.append({"role": m.role, "content": _flatten(m.content)})

            try:
                stream = await self.async_client.chat_completion(
                    messages=flat_messages,
                    model=model_id,
                    stream=True,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature or 0.7
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    tool_calls = getattr(delta, "tool_calls", None)

                    if content is None and tool_calls is None:
                        continue

                    yield ChatCompletionResponse(
                        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
                        provider='huggingface',
                        model=model_id,
                        usage={},
                        raw_response=chunk
                    )
            except Exception as e:
                # Fallback to text_generation stream
                if "Not Found" in str(e) or "conversational" in str(e):
                    last_user_msg = next((_flatten(m.content) for m in reversed(request.messages) if m.role == "user"), "")
                    async for token in await self.async_client.text_generation(
                        prompt=last_user_msg,
                        model=model_id,
                        max_new_tokens=request.max_tokens or 1024,
                        temperature=request.temperature or 0.7,
                        stream=True,
                        **provider_specific_kwargs
                    ):
                        yield ChatCompletionResponse(
                            message=ChatMessage(role="assistant", content=token),
                            provider='huggingface',
                            model=model_id,
                            usage={},
                            raw_response={"content": token}
                        )
                    return
                raise

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', None)
                response_body = getattr(e.response, 'text', str(e))
            raise map_provider_error("huggingface", e, status_code=status_code, response_body=response_body)
