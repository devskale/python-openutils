"""
HuggingFace Inference provider implementation.
"""
from typing import Dict, Any, Iterator, Optional, List

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import AuthenticationError, map_provider_error

try:
    from huggingface_hub import InferenceClient, HfApi
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

        # Initialize the HuggingFace clients
        self.client = InferenceClient(token=self.api_key)
        self.hf_api = HfApi(token=self.api_key)

    def _get_pipeline_tag(self, model_id: str) -> Optional[str]:
        """
        Retrieve the pipeline tag for the given model.

        Args:
            model_id (str): The model repository id on Hugging Face Hub.

        Returns:
            Optional[str]: The pipeline tag if available.
        """
        try:
            info = self.hf_api.model_info(model_id)
            return getattr(info, "pipeline_tag", None)
        except Exception:
            return None

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to HuggingFace Inference API.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional HuggingFace-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("HuggingFace API key is required")

        if not request.model:
            raise ValueError(
                "Model must be specified for HuggingFace Inference")

        # Normalize model id: drop pipeline tag (space) and router suffix (colon)
        base = request.model.split()[0]
        model_id = base.split(":", 1)[0]

        try:
            # Format messages for the text_generation API and flatten content arrays
            def _flatten(content: Any) -> str:
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    return "".join(parts)
                return content if isinstance(content, str) else str(content) if content is not None else ""

            last_message = None
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_message = _flatten(msg.content)
                    break

            if last_message is None:
                raise ValueError("No user message found in the request")

            # Real tool use: forwarded to HF when using chat completion

            pipeline = self._get_pipeline_tag(model_id)
            if pipeline == "conversational":
                flat_messages = []
                for m in request.messages:
                    flat_content = _flatten(m.content)
                    flat_messages.append({"role": m.role, "content": flat_content})
                extra = {}
                if "SmolLM3" in model_id or "SmolLM" in model_id:
                    extra["xml_tools"] = request.tools
                    extra["python_tools"] = request.tools
                try:
                    resp = self.client.chat_completion(messages=flat_messages, model=model_id, tools=request.tools, tool_choice=request.tool_choice, **extra)
                except Exception as e:
                    if "401" in str(e) or "Unauthorized" in str(e):
                        raise AuthenticationError("HuggingFace authentication failed for chat completion")
                    raise
                assistant_msg = getattr(resp.choices[0], "message", None)
                tool_calls = getattr(assistant_msg, "tool_calls", None) if assistant_msg else None
                if tool_calls:
                    message = ChatMessage(role="assistant", content=None, tool_calls=tool_calls)
                    completion = ""
                else:
                    assistant_content = assistant_msg.content if assistant_msg and hasattr(assistant_msg, "content") else str(resp)
                    try:
                        import re, json as _json
                        m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", assistant_content)
                        if m:
                            payload = m.group(1).strip()
                            data = _json.loads(payload)
                            tc = [{"id": f"call_{data.get('name')}", "type": "function", "function": {"name": data.get("name"), "arguments": _json.dumps(data.get("arguments", {}))}}]
                            message = ChatMessage(role="assistant", content=None, tool_calls=tc)
                            completion = ""
                        else:
                            m2 = re.search(r"<code>([\s\S]*?)</code>", assistant_content)
                            if m2:
                                code = m2.group(1)
                                fnm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", code)
                                args = {}
                                if fnm:
                                    name = fnm.group(1)
                                    argstrm = re.search(r"\((.*)\)", code)
                                    if argstrm:
                                        argstr = argstrm.group(1)
                                        for part in [p.strip() for p in argstr.split(',') if p.strip()]:
                                            kv = part.split('=', 1)
                                            if len(kv) == 2:
                                                k = kv[0].strip()
                                                v = kv[1].strip().strip('"').strip("'")
                                                args[k] = v
                                tc = [{"id": f"call_{fnm.group(1) if fnm else 'tool'}", "type": "function", "function": {"name": fnm.group(1) if fnm else "tool", "arguments": _json.dumps(args)}}]
                                message = ChatMessage(role="assistant", content=None, tool_calls=tc)
                                completion = ""
                            else:
                                message = ChatMessage(role="assistant", content=assistant_content)
                                completion = assistant_content
                    except Exception:
                        message = ChatMessage(role="assistant", content=assistant_content)
                        completion = assistant_content
                    message = ChatMessage(role="assistant", content=assistant_content)
                    completion = assistant_content
            else:
                try:
                    completion = self.client.text_generation(prompt=last_message, model=model_id, max_new_tokens=request.max_tokens or 1024, temperature=request.temperature or 0.7, **provider_specific_kwargs)
                    message = ChatMessage(role="assistant", content=completion)
                except Exception as e:
                    # Fallback to chat when text generation is unsupported
                    flat_messages = []
                    for m in request.messages:
                        flat_content = _flatten(m.content)
                        flat_messages.append({"role": m.role, "content": flat_content})
                    extra = {}
                    if "SmolLM3" in model_id or "SmolLM" in model_id:
                        extra["xml_tools"] = request.tools
                        extra["python_tools"] = request.tools
                    try:
                        resp = self.client.chat_completion(messages=flat_messages, model=model_id, tools=request.tools, tool_choice=request.tool_choice, **extra)
                    except Exception as ce:
                        if "401" in str(ce) or "Unauthorized" in str(ce):
                            raise AuthenticationError("HuggingFace authentication failed for chat completion")
                        raise
                    assistant_msg = getattr(resp.choices[0], "message", None)
                    tool_calls = getattr(assistant_msg, "tool_calls", None) if assistant_msg else None
                    if tool_calls:
                        message = ChatMessage(role="assistant", content=None, tool_calls=tool_calls)
                        completion = ""
                    else:
                        assistant_content = assistant_msg.content if assistant_msg and hasattr(assistant_msg, "content") else str(resp)
                        message = ChatMessage(role="assistant", content=assistant_content)
                        completion = assistant_content

            # Create simple usage information (HuggingFace doesn't provide detailed usage)
            usage = {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(completion.split()),
                "total_tokens": len(last_message.split()) + len(completion.split())
            }

            # Create raw response data
            raw_response = {
                "model": model_id,
                "choices": [
                    {"message": {"role": "assistant", "content": completion}}
                ],
                "usage": usage
            }

            return ChatCompletionResponse(
                message=message,
                provider='huggingface',
                model=model_id,
                usage=usage,
                raw_response=raw_response
            )
        except StopIteration as e:
            raise RuntimeError(
                f"No available inference provider for model {request.model}. "
                "Please check your setup and ensure the model or provider is supported."
            ) from e
        except Exception as e:
            if "too large to be loaded automatically" in str(e):
                model_size = str(e).split("(")[1].split(
                    ">")[0].strip() if "(" in str(e) else "unknown size"
                raise ValueError(
                    f"Model {request.model} is too large for automatic loading ({model_size} > 10GB). "
                    f"Please use one of these smaller models: {', '.join(self.list_models(self.api_key)[:5])}\n"
                    "For large models, you need to deploy them separately using Inference Endpoints: "
                    "https://huggingface.co/docs/inference-endpoints/index"
                ) from e
            elif "403 Forbidden" in str(e):
                raise PermissionError(
                    f"Access denied to model {request.model}. "
                    "Possible reasons:\n"
                    "1. Your API key lacks permissions for this model\n"
                    "2. The model requires gated access (may need application)\n"
                    "3. The model is private\n\n"
                    "Check permissions at: https://huggingface.co/settings/tokens\n"
                    f"Request access at: https://huggingface.co/{request.model.split('/')[0]}"
                ) from e
            elif "Pro subscription" in str(e):
                raise PermissionError(
                    f"Model {request.model} requires a Pro subscription. \n"
                    "You can either:\n"
                    "1. Upgrade your account at https://huggingface.co/pricing\n"
                    f"2. Use one of these free models: {', '.join(self.list_models(self.api_key)[:5])}"
                ) from e
            status_code = None
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', None)
                response_body = getattr(e.response, 'text', str(e))
            raise map_provider_error("huggingface", e, status_code=status_code, response_body=response_body)

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> List[str]:
        """
        List available models from HuggingFace.

        Args:
            api_key (Optional[str]): The HuggingFace API key.

        Returns:
            List[str]: A list of available model names.

        Raises:
            Exception: If the request fails.
        """
        try:
            if not HAS_HUGGINGFACE:
                raise ImportError(
                    "huggingface_hub package is required for the HuggingFaceProvider. "
                    "Install it with: pip install huggingface_hub"
                )

            if not api_key:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("huggingface")
                if not api_key:
                    raise ValueError(
                        "HuggingFace API key is required for listing models")

            # Initialize HfApi client
            hf_api = HfApi(token=api_key)

            # Fetch text-generation models sorted by popularity
            models = hf_api.list_models(
                pipeline_tag="text-generation",
                sort="likes",
                limit=100
            )
            # Create a list of model names
            model_list = []
            for model in models:
                if model.id:
                    # drop the "text-generation" tag from the name
                    if model.pipeline_tag and model.pipeline_tag != "text-generation":
                        model_list.append(f"{model.id} {model.pipeline_tag}")
                    else:
                        model_list.append(model.id)
            # Extract model IDs
            return model_list
        except Exception as e:
            # Log the error for debugging
            import logging
            logging.warning(f"Failed to fetch HuggingFace models: {str(e)}")

            # Fallback to default models if API call fails
            return [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-70B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "google/gemma-7b-it"
            ]

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from HuggingFace Inference API.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional HuggingFace-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("HuggingFace API key is required")

        if not request.model:
            raise ValueError(
                "Model must be specified for HuggingFace Inference")

        # Normalize model id: drop pipeline tag (space) and router suffix (colon)
        base = request.model.split()[0]
        model_id = base.split(":", 1)[0]

        try:
            # Flatten content arrays for streaming
            def _flatten(content: Any) -> str:
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    return "".join(parts)
                return content if isinstance(content, str) else str(content) if content is not None else ""

            last_message = None
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_message = _flatten(msg.content)
                    break

            if last_message is None:
                raise ValueError("No user message found in the request")

            # Tool use handled via chat stream when pipeline is conversational or on fallback

            pipeline = self._get_pipeline_tag(model_id)
            if pipeline == "conversational":
                flat_messages = []
                for m in request.messages:
                    flat_content = _flatten(m.content)
                    flat_messages.append({"role": m.role, "content": flat_content})
                extra = {}
                if "SmolLM3" in model_id or "SmolLM" in model_id:
                    extra["xml_tools"] = request.tools
                    extra["python_tools"] = request.tools
                try:
                    chat_stream = self.client.chat_completion(messages=flat_messages, model=model_id, stream=True, tools=request.tools, tool_choice=request.tool_choice, **extra)
                except Exception as e:
                    if "401" in str(e) or "Unauthorized" in str(e):
                        raise AuthenticationError("HuggingFace authentication failed for chat completion")
                    raise
                tc_active = False
                tc_buf = ""
                for chunk in chat_stream:
                    delta_content = None
                    delta_tools = None
                    try:
                        if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta"):
                            delta_obj = chunk.choices[0].delta
                            delta_content = getattr(delta_obj, "content", None)
                            delta_tools = getattr(delta_obj, "tool_calls", None)
                    except Exception:
                        delta_content = None
                        delta_tools = None
                    emitted = False
                    try:
                        import re, json as _json
                        if delta_content:
                            if not tc_active and "<tool_call>" in delta_content:
                                tc_active = True
                                tc_buf = ""
                            if tc_active:
                                tc_buf += delta_content
                                if "</tool_call>" in delta_content:
                                    tc_active = False
                                    m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", tc_buf)
                                    if m:
                                        payload = m.group(1).strip()
                                        data = _json.loads(payload)
                                        tc = [{"id": f"call_{data.get('name')}", "type": "function", "function": {"name": data.get("name"), "arguments": _json.dumps(data.get("arguments", {}))}}]
                                        message = ChatMessage(role="assistant", content=None, tool_calls=tc)
                                        raw_response = {"model": model_id, "choices": [{"delta": {"tool_calls": tc}}]}
                                        yield ChatCompletionResponse(message=message, provider='huggingface', model=model_id, usage={}, raw_response=raw_response)
                                        emitted = True
                                        tc_buf = ""
                        if not emitted and delta_tools:
                            message = ChatMessage(role="assistant", content=None, tool_calls=delta_tools)
                            raw_response = {"model": model_id, "choices": [{"delta": {"tool_calls": delta_tools}}]}
                            yield ChatCompletionResponse(message=message, provider='huggingface', model=model_id, usage={}, raw_response=raw_response)
                            emitted = True
                    except Exception:
                        pass
                    if not emitted:
                        if delta_content is None and not delta_tools:
                            continue
                        message = ChatMessage(role="assistant", content=delta_content, tool_calls=delta_tools)
                        raw_delta = {}
                        if delta_content is not None:
                            raw_delta["content"] = delta_content
                        if delta_tools:
                            raw_delta["tool_calls"] = delta_tools
                        raw_response = {"model": model_id, "choices": [{"delta": raw_delta}]}
                        yield ChatCompletionResponse(message=message, provider='huggingface', model=model_id, usage={}, raw_response=raw_response)
            else:
                try:
                    stream = self.client.text_generation(prompt=last_message, model=model_id, max_new_tokens=request.max_tokens or 500, temperature=request.temperature or 0.7, stream=True, **provider_specific_kwargs)
                    for chunk in stream:
                        message = ChatMessage(role="assistant", content=chunk)
                        raw_response = {"model": model_id, "choices": [{"delta": {"content": chunk}}]}
                        yield ChatCompletionResponse(message=message, provider='huggingface', model=model_id, usage={}, raw_response=raw_response)
                except Exception:
                    # Fallback to chat stream when text generation is unsupported
                    flat_messages = []
                    for m in request.messages:
                        flat_content = _flatten(m.content)
                        flat_messages.append({"role": m.role, "content": flat_content})
                    chat_stream = self.client.chat_completion(messages=flat_messages, model=model_id, stream=True)
                    for chunk in chat_stream:
                        delta = None
                        try:
                            delta = chunk.choices[0].delta.content if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta") else None
                        except Exception:
                            delta = None
                        if not delta:
                            continue
                        message = ChatMessage(role="assistant", content=delta)
                        raw_response = {"model": model_id, "choices": [{"delta": {"content": delta}}]}
                        yield ChatCompletionResponse(message=message, provider='huggingface', model=model_id, usage={}, raw_response=raw_response)
        except StopIteration as e:
            raise RuntimeError(
                f"No available inference provider for model {request.model}. "
                "Please check your setup and ensure the model or provider is supported."
            ) from e
        except Exception as e:
            if "too large to be loaded automatically" in str(e):
                model_size = str(e).split("(")[1].split(
                    ">")[0].strip() if "(" in str(e) else "unknown size"
                raise ValueError(
                    f"Model {request.model} is too large for automatic loading ({model_size} > 10GB). "
                    f"Please use one of these smaller models: {', '.join(self.list_models(self.api_key)[:5])}\n"
                    "For large models, you need to deploy them separately using Inference Endpoints: "
                    "https://huggingface.co/docs/inference-endpoints/index"
                ) from e
            elif "403 Forbidden" in str(e):
                raise PermissionError(
                    f"Access denied to model {request.model}. "
                    "Possible reasons:\n"
                    "1. Your API key lacks permissions for this model\n"
                    "2. The model requires gated access (may need application)\n"
                    "3. The model is private\n\n"
                    "Check permissions at: https://huggingface.co/settings/tokens\n"
                    f"Request access at: https://huggingface.co/{request.model.split('/')[0]}"
                ) from e
            elif "Pro subscription" in str(e):
                raise PermissionError(
                    f"Model {request.model} requires a Pro subscription. \n"
                    "You can either:\n"
                    "1. Upgrade your account at https://huggingface.co/pricing\n"
                    f"2. Use one of these free models: {', '.join(self.list_models(self.api_key)[:5])}"
                ) from e
            status_code = None
            response_body = str(e)
            if hasattr(e, 'response') and e.response is not None:
                status_code = getattr(e.response, 'status_code', None)
                response_body = getattr(e.response, 'text', str(e))
            raise map_provider_error("huggingface", e, status_code=status_code, response_body=response_body)
