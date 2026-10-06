from __future__ import annotations
"""
Google Gemini provider implementation with async support.
"""
import base64
import json
from typing import Dict, Any, Optional, List, AsyncIterator

from ..core import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatProvider, ModelInfo
from ..errors import map_provider_error, UniInferError, RateLimitError

# Try to import google-genai package (latest recommended package)
# Note: Install with 'uv pip install google-genai' if not available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _parse_gemini_quota(error: Exception) -> dict:
    """Extract quota info from a google-genai ClientError (429).

    The error message typically contains text like:
      "Quota exceeded for quota metric 'GenerateRequestsPerDayPerModel-FreeTier' with limit '20'"

    Returns a dict with optional 'quota_metric' and 'quota_limit' keys.
    """
    info = {}

    message = getattr(error, 'message', None) or str(error)

    import re
    metric_match = re.search(r"quota metric '([^']+)'", message)
    if metric_match:
        info['quota_metric'] = metric_match.group(1)

    limit_match = re.search(r"limit '(\d+)'", message)
    if limit_match:
        info['quota_limit'] = int(limit_match.group(1))

    return info


def _sanitize_gemini_schema(node: Any) -> Any:
    """Strip schema constructs the live Gemini API rejects with 400 INVALID_ARGUMENT.

    The google-genai SDK happily accepts OpenAI-style keywords (it has fields
    for them), but the API rejects the serialized payload: ``additionalProperties``
    (true or false) is unknown at every schema position, including inside
    ``anyOf`` branches (verified against the live API 2026-11). Drop it
    recursively; an open object stays open by omission, a closed one loses
    only a constraint the API cannot express anyway.
    """
    if isinstance(node, dict):
        out = {k: _sanitize_gemini_schema(v) for k, v in node.items() if k != "additionalProperties"}
        return out
    if isinstance(node, list):
        return [_sanitize_gemini_schema(v) for v in node]
    return node


class GeminiProvider(ChatProvider):
    """
    Provider for Google Gemini API with async support.
    
    Uses the modern google-genai SDK which is natively async.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Gemini provider.

        Args:
            api_key (Optional[str]): The Gemini API key.
            **kwargs: Additional provider-specific configuration parameters.
        """
        resolved_key = api_key
        if not resolved_key:
            try:
                from credgoo import get_api_key
                resolved_key = get_api_key("gemini")
            except (ImportError, Exception):
                pass

        super().__init__(resolved_key)
        self._client = None

        if not HAS_GENAI:
            raise ImportError(
                "The 'google-genai' package is required to use the Gemini provider. "
                "Install it with 'uv pip install google-genai'"
            )

        self.config = kwargs

    # Module-level client cache: the proxy instantiates a fresh provider per
    # request, so an instance-level cache never hits. genai.Client construction
    # builds an httpx client incl. ssl.create_default_context (blocking, loads
    # system CAs) — doing that per request ON THE EVENT LOOP wedged the whole
    # proxy under sustained load (2026-09-21 smoke sweep). One client per api_key
    # is safe: genai clients are stateless beyond the transport.
    _CLIENT_CACHE: dict[str, Any] = {}

    def _get_client(self):
        """
        Get or create the Gemini client, cached per api_key at module level.
        A 120s http timeout is baked in — without it a hung upstream call
        blocks its request forever (aio has no default timeout).
        """
        cached = GeminiProvider._CLIENT_CACHE.get(self.api_key)
        if cached is None:
            cached = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": 120_000},  # ms
            )
            GeminiProvider._CLIENT_CACHE[self.api_key] = cached
        self._client = cached
        return cached

    async def aclose(self):
        """
        Close the Gemini client.
        """
        if self._client is not None:
            # The new google-genai Client doesn't have an explicit close for sync, 
            # but we can clear our reference. Async client is accessed via aio.
            self._client = None
        await super().aclose()

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list[ModelInfo]:
        """
        List available models from Gemini.

        Args:
            api_key (Optional[str]): The Gemini API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list[ModelInfo]: A list of model info objects.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        if not api_key:
            try:
                from credgoo import get_api_key
                api_key = get_api_key('gemini')
            except (ImportError, Exception):
                pass

        if not api_key:
            raise ValueError("Gemini API key is required for listing models")

        try:
            client = genai.Client(api_key=api_key)
            results = []

            for model in client.models.list():
                model_name = getattr(model, 'name', None)
                if not model_name:
                    continue

                capabilities = {}
                if getattr(model, 'thinking', None):
                    capabilities["thinking"] = True

                supported_actions = getattr(model, 'supported_actions', None) or []
                if supported_actions:
                    capabilities["supported_actions"] = supported_actions

                # Strip the "models/" prefix the API puts on model names so
                # catalog/serving ids are clean (e.g. gemini-2.5-flash).
                clean_id = model_name.removeprefix("models/")
                results.append(ModelInfo(
                    id=clean_id,
                    name=getattr(model, 'display_name', None),
                    type="chat",
                    context_window=getattr(model, 'input_token_limit', None),
                    max_output=getattr(model, 'output_token_limit', None),
                    capabilities=capabilities or None,
                    raw=model.model_dump() if hasattr(model, 'model_dump') else None,
                ))

            return results

        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            raise map_provider_error("Gemini", e, status_code=status_code, response_body=response_body)

    def _prepare_content_and_config(self, request: ChatCompletionRequest) -> tuple:
        """
        Prepare content and config for Gemini API from our messages.

        Args:
            request (ChatCompletionRequest): The request to prepare for.

        Returns:
            tuple: (content, config, tools) for Gemini API.
        """
        # Extract all messages
        messages = request.messages

        def _flatten_text(content: Any) -> str:
            """
            Normalize message content to plain text.
            Supports OpenAI-style content arrays: [{"type":"text","text":"..."}, ...].
            """
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                if parts:
                    return "".join(parts)
                # Fallback: join any string-like items
                return "".join(str(p) for p in content)
            return content if isinstance(content, str) else str(content) if content is not None else ""

        # Look for system message
        system_message = None
        for msg in messages:
            if msg.role == "system":
                system_message = _flatten_text(msg.content)
                break

        # Prepare config with generation parameters
        config_params = {}
        if request.temperature is not None:
            config_params["temperature"] = request.temperature
        if request.max_tokens is not None:
            config_params["max_output_tokens"] = request.max_tokens
        if request.tools:
            tool_choice = request.tool_choice
            if tool_choice is None or tool_choice == "auto":
                config_params["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                )
            elif tool_choice == "required":
                config_params["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )
            elif tool_choice == "none":
                config_params["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                )
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                config_params["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[tool_choice["function"]["name"]]
                    )
                )

        # Create a config object using new types structure
        config = types.GenerateContentConfig(**config_params)

        # Prepare content based on non-system messages
        gemini_tools = None

        # For simple queries with just one user message, use a simple string
        if len(messages) == 1 and messages[0].role == "user":
            content = _flatten_text(messages[0].content)
        else:
            # Map each message to its Gemini shape (2026-11):
            # user -> user turn (text); assistant -> model turn, with tool_calls
            # as functionCall parts (args must be an object, not a JSON string);
            # tool results -> functionResponse parts on a USER turn, keyed by
            # the function NAME. The previous flat mapping (everything
            # non-user -> model text parts) silently dropped function calls and
            # sent tool results as bare model text.
            def _tool_name_for_call_id(call_id):
                for m in messages:
                    for tc in (m.tool_calls or []):
                        if tc.get("id") == call_id:
                            return (tc.get("function") or {}).get("name")
                return None

            def _args_object(raw):
                """OpenAI tool arguments (JSON string) -> Gemini args object."""
                if isinstance(raw, dict):
                    return raw
                if isinstance(raw, str) and raw.strip():
                    try:
                        value = json.loads(raw)
                        return value if isinstance(value, dict) else {"value": value}
                    except (json.JSONDecodeError, ValueError):
                        return {"raw": raw}
                return {}

            def _result_object(raw):
                """Tool-result content -> object for functionResponse.response."""
                text = _flatten_text(raw)
                if text.strip():
                    try:
                        value = json.loads(text)
                        return value if isinstance(value, dict) else {"result": value}
                    except (json.JSONDecodeError, ValueError):
                        return {"result": text}
                return {"result": ""}

            def _signature_map(msg):
                """tool_call id -> base64 thought signature (Gemini 3 requires the
                original signature replayed on every functionCall part; without
                it the API 400s 'Function call is missing a thought_signature').
                Carried through the OpenAI-compat wire as reasoning.encrypted
                reasoning_details keyed by the tool_call id."""
                sigs = {}
                for d in getattr(msg, "reasoning_details", None) or []:
                    if isinstance(d, dict) and d.get("type") == "reasoning.encrypted" and d.get("id") and d.get("data"):
                        sigs[d["id"]] = d["data"]
                return sigs

            content = []
            for msg in messages:
                if msg.role == "system":
                    continue
                if msg.role == "tool":
                    name = _tool_name_for_call_id(msg.tool_call_id) or "unknown_function"
                    content.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": name,
                            "response": _result_object(msg.content),
                        }}],
                    })
                    continue
                role = "user" if msg.role == "user" else "model"
                parts = []
                if role == "model" and msg.tool_calls:
                    if msg.content:
                        parts.append({"text": _flatten_text(msg.content)})
                    signatures = _signature_map(msg)
                    for tc in msg.tool_calls:
                        fn = tc.get("function") or {}
                        call_part = {"functionCall": {
                            "name": fn.get("name"),
                            "args": _args_object(fn.get("arguments")),
                        }}
                        signature = signatures.get(tc.get("id"))
                        if signature:
                            call_part["thoughtSignature"] = signature
                        parts.append(call_part)
                else:
                    parts.append({"text": _flatten_text(msg.content)})
                content.append({"role": role, "parts": parts})

            # Gemini rejects transcripts whose final turn is a model turn
            # ("Requests ending with a model turn are not supported" —
            # prefill-style transcripts, enforced by newer models like
            # gemini-3.5-flash-lite; older models tolerated them). Close with a
            # neutral user turn so generation has a turn to respond to.
            if content and content[-1]["role"] == "model":
                content.append({"role": "user", "parts": [{"text": "Continue."}]})

        # Convert OpenAI tools format to Gemini function declarations
        if request.tools:
            gemini_tools = []
            for tool in request.tools:
                if tool.get('type') == 'function':
                    func = tool.get('function', {})
                    gemini_func = {
                        "name": func.get('name'),
                        "description": func.get('description', ''),
                    }
                    if 'parameters' in func:
                        gemini_func["parameters"] = _sanitize_gemini_schema(func['parameters'])
                    gemini_tools.append(gemini_func)

        if system_message:
            # Prepend system message to the first user message, or add it as the first message
            if isinstance(content, list) and content and 'role' in content[0] and content[0]['role'] == 'user':
                if isinstance(content[0]['parts'], list) and 'text' in content[0]['parts'][0]:
                    content[0]['parts'][0]['text'] = f"{system_message}\n{content[0]['parts'][0]['text']}"
                else:
                    # Fallback if structure is not as expected
                    content.insert(0, {"role": "user", "parts": [
                                    {"text": system_message}]})
            elif isinstance(content, str):
                # If content is a string, it means it's a single user message
                # Convert it to the expected format for list content if we need to add a system message
                content = [{"role": "user", "parts": [{"text": f"{system_message}\n{content}"}]}]
            else:
                # If no user message or other format, add system message as user message
                if not isinstance(content, list):
                    content = []
                content.insert(0, {"role": "user", "parts": [
                                {"text": system_message}]})

        return content, config, gemini_tools

    def _prepare_api_params(
        self,
        request: ChatCompletionRequest,
        content: Any,
        config: Any,
        gemini_tools: Optional[List[Dict]],
        **provider_specific_kwargs
    ) -> Dict[str, Any]:
        """Prepare API parameters for Gemini."""
        model = request.model or "gemini-1.5-flash"
        api_params = {
            "model": model,
            "contents": content,
            "config": config
        }

        if gemini_tools:
            from google.genai.types import Tool, FunctionDeclaration
            tool_declarations = []
            for func_def in gemini_tools:
                tool_declarations.append(
                    FunctionDeclaration(
                        name=func_def["name"],
                        description=func_def.get("description", ""),
                        parameters=func_def.get("parameters")
                    )
                )
            existing = config.model_dump(exclude_none=True)
            existing["tools"] = [Tool(function_declarations=tool_declarations)]
            api_params["config"] = types.GenerateContentConfig(**existing)

        api_params.update(provider_specific_kwargs)
        return api_params

    def _extract_usage_and_finish_reason(self, response: Any) -> tuple[dict, str | None]:
        """Extract usage and finish_reason from a Gemini response."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason = None

        # Usage from response.usage_metadata
        um = getattr(response, 'usage_metadata', None)
        if um:
            usage["prompt_tokens"] = getattr(um, 'promptTokenCount', 0) or 0
            usage["completion_tokens"] = getattr(um, 'candidatesTokenCount', 0) or 0
            usage["total_tokens"] = getattr(um, 'totalTokenCount', 0) or 0
            thoughts_tokens = getattr(um, 'thoughtsTokenCount', 0) or 0
            if thoughts_tokens:
                usage["completion_tokens_details"] = {"reasoning_tokens": thoughts_tokens}

        # Finish reason from first candidate
        candidates = getattr(response, 'candidates', None)
        if candidates:
            fr = getattr(candidates[0], 'finish_reason', None)
            if fr:
                finish_reason = fr.name if hasattr(fr, 'name') else str(fr)

        return usage, finish_reason

    def _process_response_common(
        self,
        response: Any,
        model: str,
        request_model: Optional[str]
    ) -> ChatCompletionResponse:
        """Process a Gemini API response into a ChatCompletionResponse."""
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise UniInferError(
                f"Gemini content generation blocked. Reason: {response.prompt_feedback.block_reason}. "
                f"Safety ratings: {response.prompt_feedback.safety_ratings}"
            )

        content_text = ""
        tool_calls = None

        thinking_content = None
        if response.parts:
            # Collect thinking and content parts
            thinking_parts = []
            text_parts = []
            
            for part in response.parts:
                # Gemini 2.0 Thinking models use the 'thought' attribute on parts
                if hasattr(part, 'thought') and part.thought:
                    if hasattr(part, 'text') and part.text:
                        thinking_parts.append(part.text)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            if thinking_parts:
                thinking_content = "".join(thinking_parts)
                
            if text_parts:
                content_text = "".join(text_parts)
            else:
                # Try the standard .text property if manual extraction produced nothing
                try:
                    content_text = response.text
                except (ValueError, AttributeError):
                    content_text = ""

            function_calls = []
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append({
                        "id": f"call_{part.function_call.name}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args))
                        }
                    })
            if function_calls:
                tool_calls = function_calls

        usage, finish_reason = self._extract_usage_and_finish_reason(response)

        # Gemini returns STOP even when function calls are present
        if tool_calls and finish_reason != "tool_calls":
            finish_reason = "tool_calls"

        message = ChatMessage(
            role="assistant",
            content=content_text if content_text else None,
            tool_calls=tool_calls
        )

        return ChatCompletionResponse(
            message=message,
            provider='gemini',
            model=model,
            usage=usage,
            raw_response=response,
            thinking=thinking_content,
            reasoning_details=self._extract_reasoning_details(response),
            finish_reason=finish_reason
        )

    def _extract_reasoning_details(self, response: Any) -> Optional[List[dict]]:
        """Capture Gemini 3 thought signatures as reasoning.encrypted details.

        The signature bytes ride on the same Part as the functionCall; they are
        opaque and MUST come back verbatim on the next request, or the API
        rejects the tool round (missing/corrupted thought_signature). Wire
        shape is what pi/openrouter-style clients preserve across turns:
        {type: reasoning.encrypted, id: <tool_call id>, data: <base64>}.
        """
        details = []
        for index, part in enumerate(getattr(response, 'parts', None) or []):
            fc = getattr(part, 'function_call', None)
            signature = getattr(part, 'thought_signature', None)
            if fc and isinstance(signature, (bytes, bytearray)):
                details.append({
                    "type": "reasoning.encrypted",
                    "id": f"call_{fc.name}",
                    "data": base64.b64encode(signature).decode("ascii"),
                    "format": "gemini",
                    "index": index,
                })
        return details or None

    def _complete_impl(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Internal implementation of synchronous completion for Gemini."""
        client = self._get_client()

        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            content, config, gemini_tools = self._prepare_content_and_config(request)
            api_params = self._prepare_api_params(request, content, config, gemini_tools, **provider_specific_kwargs)
            response = client.models.generate_content(**api_params)
            return self._process_response_common(response, api_params['model'], request.model)

        except Exception as e:
            status_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            response_body = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_body = e.response.text
                status_code = e.response.status_code
            elif hasattr(e, 'message'):
                response_body = e.message

            mapped_error = map_provider_error("gemini", e, status_code=status_code, response_body=response_body)

            if status_code == 429 and isinstance(mapped_error, RateLimitError):
                quota = _parse_gemini_quota(e)
                mapped_error.quota_metric = quota.get('quota_metric')
                mapped_error.quota_limit = quota.get('quota_limit')

            raise mapped_error

    async def _acomplete_impl(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Internal implementation of asynchronous completion for Gemini."""
        # First construction per process does blocking work (SSL/CA loading) —
        # keep it off the event loop; afterwards the cache hits are cheap.
        if self.api_key not in GeminiProvider._CLIENT_CACHE:
            import asyncio
            client = await asyncio.to_thread(self._get_client)
        else:
            client = self._get_client()

        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            content, config, gemini_tools = self._prepare_content_and_config(request)
            api_params = self._prepare_api_params(request, content, config, gemini_tools, **provider_specific_kwargs)
            response = await client.aio.models.generate_content(**api_params)
            return self._process_response_common(response, api_params['model'], request.model)

        except Exception as e:
            status_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            response_body = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_body = e.response.text
                status_code = e.response.status_code
            elif hasattr(e, 'message'):
                response_body = e.message

            mapped_error = map_provider_error("gemini", e, status_code=status_code, response_body=response_body)

            if status_code == 429 and isinstance(mapped_error, RateLimitError):
                quota = _parse_gemini_quota(e)
                mapped_error.quota_metric = quota.get('quota_metric')
                mapped_error.quota_limit = quota.get('quota_limit')

            raise mapped_error

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """Make an async chat completion request to Gemini."""
        return await self._acomplete_impl(request, **provider_specific_kwargs)

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """Stream an async chat completion response from Gemini."""
        client = self._get_client()

        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            content, config, gemini_tools = self._prepare_content_and_config(request)
            api_params = self._prepare_api_params(request, content, config, gemini_tools, **provider_specific_kwargs)

            model = api_params['model']
            async for chunk in await client.aio.models.generate_content_stream(**api_params):
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    raise UniInferError(
                        f"Gemini content generation blocked. Reason: {chunk.prompt_feedback.block_reason}. "
                        f"Safety ratings: {chunk.prompt_feedback.safety_ratings}"
                    )

                content_text = ""
                tool_calls = None
                
                if chunk.parts:
                    thinking_parts = []
                    text_parts = []
                    
                    for part in chunk.parts:
                        # Gemini 2.0 Thinking models use the 'thought' attribute on parts
                        if hasattr(part, 'thought') and part.thought:
                            if hasattr(part, 'text') and part.text:
                                thinking_parts.append(part.text)
                        elif hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    thinking_content_chunk = "".join(thinking_parts) if thinking_parts else None
                    
                    if text_parts:
                        content_text = "".join(text_parts)
                    else:
                        # Try the standard .text property as fallback
                        try:
                            content_text = chunk.text
                        except (ValueError, AttributeError):
                            content_text = ""

                    # Extract function calls if present
                    function_calls = []
                    for part in chunk.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append({
                                "id": f"call_{part.function_call.name}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(dict(part.function_call.args))
                                }
                            })
                    if function_calls:
                        tool_calls = function_calls

                if content_text or tool_calls or thinking_content_chunk:
                    message = ChatMessage(
                        role="assistant", 
                        content=content_text if content_text else None,
                        tool_calls=tool_calls
                    )
                    usage, finish_reason = self._extract_usage_and_finish_reason(chunk)
                    if tool_calls and finish_reason != "tool_calls":
                        finish_reason = "tool_calls"
                    yield ChatCompletionResponse(
                        message=message,
                        provider='gemini',
                        model=model,
                        usage=usage,
                        raw_response=chunk,
                        thinking=thinking_content_chunk,
                        reasoning_details=self._extract_reasoning_details(chunk),
                        finish_reason=finish_reason
                    )

        except Exception as e:
            if isinstance(e, UniInferError):
                raise
            status_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            response_body = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_body = e.response.text
                status_code = e.response.status_code
            elif hasattr(e, 'message'):
                response_body = e.message

            mapped_error = map_provider_error("gemini", e, status_code=status_code, response_body=response_body)

            if status_code == 429 and isinstance(mapped_error, RateLimitError):
                quota = _parse_gemini_quota(e)
                mapped_error.quota_metric = quota.get('quota_metric')
                mapped_error.quota_limit = quota.get('quota_limit')

            raise mapped_error
