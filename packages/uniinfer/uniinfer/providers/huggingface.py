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
        if hasattr(self, "async_client"):
            try:
                # Try to close the async client if it has a close method
                if hasattr(self.async_client, "close"):
                    await self.async_client.close()
            except Exception:
                pass
        await super().aclose()

    def _get_pipeline_tag(self, model_id: str) -> Optional[str]:
        """Retrieve the pipeline tag for the given model."""
        try:
            info = self.hf_api.model_info(model_id)
            return getattr(info, "pipeline_tag", None)
        except Exception:
            return None

    def _flatten_content(self, content: Any) -> str:
        """Helper to flatten message content to string."""
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content) if content is not None else ""

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
            # Filter for conversational models as they are more likely to work with chat_completion
            models = hf_api.list_models(filter="conversational", sort="downloads", limit=100)
            return [model.id for model in models if model.id]
        except Exception:
            return [
                "meta-llama/Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "google/gemma-3-27b-it",
                "mistralai/Mistral-7B-Instruct-v0.3"
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
            flat_messages = []
            for m in request.messages:
                flat_messages.append({"role": m.role, "content": self._flatten_content(m.content)})

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
                    **{**extra, **provider_specific_kwargs}
                )
            except Exception as e:
                # Smarter Fallback Logic
                error_str = str(e).lower()
                
                # If the error explicitly says the task IS supported, do NOT fallback
                if "supported task: conversational" in error_str:
                    raise e

                # Retrieve tag to make a better decision
                tag = self._get_pipeline_tag(model_id)
                is_chat_tagged = tag in ["conversational", "image-text-to-text", "image-to-text", "vlm"]
                
                # If the model name suggests it's an instruct/chat model, be very conservative
                model_is_chat_named = any(term in model_id.lower() for term in ["instruct", "chat", "-it", "it-v"])

                # Check if the error indicates a mismatch in model type
                is_explicit_non_chat = "is not a chat model" in error_str
                
                # We also fallback if the hub fails to find any provider (StopIteration/RuntimeError)
                is_routing_err = any(term in error_str for term in ["stopiteration", "runtimeerror", "coroutine raised"])

                # We ONLY fallback if:
                # 1. It's an explicit "not a chat model" error
                # 2. It's a routing error AND it doesn't look like a chat model
                should_fallback = is_explicit_non_chat or (is_routing_err and not (is_chat_tagged or model_is_chat_named))

                if should_fallback:
                    # construct a prompt from all messages
                    prompt = ""
                    for m in flat_messages:
                        prompt += f"{m['role']}: {m['content']}\n\n"
                    prompt += "assistant:"

                    try:
                        completion = await self.async_client.text_generation(
                            prompt=prompt,
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
                    except (RuntimeError, StopIteration) as hub_bug:
                        raise UniInferError(f"HuggingFace library routing error for model '{model_id}': {str(hub_bug)}. This model might not be supported on the serverless Inference API yet.") from hub_bug
                    except Exception as text_gen_err:
                        # If even text_generation fails, raise the original error or the new one
                        raise text_gen_err
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
            flat_messages = []
            for m in request.messages:
                flat_messages.append({"role": m.role, "content": self._flatten_content(m.content)})

            try:
                stream = await self.async_client.chat_completion(
                    messages=flat_messages,
                    model=model_id,
                    stream=True,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature or 0.7,
                    **provider_specific_kwargs
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    tool_calls = delta.tool_calls if hasattr(delta, "tool_calls") else None

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
                # Smarter Fallback Logic
                error_str = str(e).lower()
                
                # If the error explicitly says the task IS supported, do NOT fallback
                if "supported task: conversational" in error_str:
                    raise e

                # Retrieve tag to make a better decision
                tag = self._get_pipeline_tag(model_id)
                is_chat_tagged = tag in ["conversational", "image-text-to-text", "image-to-text", "vlm"]
                
                # If the model name suggests it's an instruct/chat model, be very conservative
                model_is_chat_named = any(term in model_id.lower() for term in ["instruct", "chat", "-it", "it-v"])

                # Check if the error indicates a mismatch in model type
                is_explicit_non_chat = "is not a chat model" in error_str
                
                # We also fallback if the hub fails to find any provider (StopIteration/RuntimeError)
                is_routing_err = any(term in error_str for term in ["stopiteration", "runtimeerror", "coroutine raised"])

                # We ONLY fallback if:
                # 1. It's an explicit "not a chat model" error
                # 2. It's a routing error AND it doesn't look like a chat model
                should_fallback = is_explicit_non_chat or (is_routing_err and not (is_chat_tagged or model_is_chat_named))

                if should_fallback:
                    # construct a prompt from all messages
                    prompt = ""
                    for m in flat_messages:
                        prompt += f"{m['role']}: {m['content']}\n\n"
                    prompt += "assistant:"

                    try:
                        text_gen_stream = await self.async_client.text_generation(
                            prompt=prompt,
                            model=model_id,
                            max_new_tokens=request.max_tokens or 1024,
                            temperature=request.temperature or 0.7,
                            stream=True,
                            **provider_specific_kwargs
                        )
                        async for token in text_gen_stream:
                            yield ChatCompletionResponse(
                                message=ChatMessage(role="assistant", content=token),
                                provider='huggingface',
                                model=model_id,
                                usage={},
                                raw_response={"content": token}
                            )
                    except (RuntimeError, StopIteration) as hub_bug:
                        raise UniInferError(f"HuggingFace library routing error for model '{model_id}': {str(hub_bug)}. This model might not be supported on the serverless Inference API yet.") from hub_bug
                    except Exception as text_gen_err:
                        raise text_gen_err
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
