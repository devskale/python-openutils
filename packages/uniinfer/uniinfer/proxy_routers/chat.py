import logging
import time
from typing import Any, AsyncGenerator, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.completion import Target
from uniinfer.config.instances import instance_requires_api_key
from uniinfer.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    UniInferError,
)
from uniinfer.provider_access import _resolve_credgoo_service, get_embeddings
from uniinfer.proxy_schemas.chat import (
    ChatCompletionRequestInput,
    ChatMessageOutput,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    NonStreamingChatCompletion,
    NonStreamingChoice,
)
from uniinfer.proxy_services.key_usage import get_key_usage
from uniinfer.proxy_services.stats import get_stats
from uniinfer.proxy_services.streaming import (
    is_openai_strict_mode,
    normalize_nonstream_content,
)

logger = logging.getLogger("uniioai_proxy")

# Providers whose native API accepts the OpenAI 'developer' message role as-is.
# Everyone else gets 'developer' collapsed to 'system' (functionally equivalent;
# universally accepted) so they don't 422. Add a provider here only once its API
# is confirmed to accept 'developer'.
_DEVELOPER_ROLE_PROVIDERS = frozenset({"openai", "openrouter"})

# Auto-routing: max fallback candidates tried after the first choice fails.
_AUTO_MAX_FALLBACKS = 3


def create_chat_router(
    *,
    parse_provider_model: Callable[..., tuple[str, str]],
    provider_configs: dict[str, Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        request_input: ChatCompletionRequestInput,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        base_url = request_input.base_url
        provider_model = request_input.model
        # The END log line (middleware) names the model — without it a mixed
        # traffic journal cannot tell which model a status belonged to.
        request.state.provider_model = provider_model
        _req_t0 = time.monotonic()
        target: Target | None = None

        try:
            _own_key = request.headers.get("x-provider-key")

            # Dispatch resolution (parse + auth + auto-route + fallback chain)
            # lives in the shared core — chat is a thin surface over it.
            from uniinfer.proxy_services.completion_dispatch import resolve_dispatch

            def _on_fallback(new_model: str) -> None:
                nonlocal provider_model, target
                request.state.provider_model = new_model
                request.state.routed_model = new_model
                provider_model = new_model
                target = plan.target

            plan = resolve_dispatch(
                provider_model,
                api_bearer_token,
                own_key=_own_key,
                base_url=base_url,
            )
            plan.on_fallback = _on_fallback
            provider_name = plan.provider_name
            provider_model = plan.provider_model
            target = plan.target
            if plan.routed_model:
                request.state.routed_model = provider_model
            request.state.provider_model = provider_model
            messages_dict = [msg.model_dump() for msg in request_input.messages]
            # OpenAI-compat: 'developer' is OpenAI's newer system-instructions role
            # (reasoning models; emitted by the official SDKs). Most backends
            # (Mistral, Gemini, Ollama, TU/vLLM, Anthropic, …) reject it with 422,
            # so collapse to 'system' (functionally equivalent, universally
            # accepted) unless the provider natively accepts 'developer'.
            if provider_name not in _DEVELOPER_ROLE_PROVIDERS:
                for _m in messages_dict:
                    if _m.get("role") == "developer":
                        _m["role"] = "system"
            # OpenAI passthrough: forward unmapped OpenAI params (top_p,
            # response_format, seed, stream_options, …) to the backend instead
            # of dropping them. The schema captures them via extra="allow".
            extras = dict(getattr(request_input, "__pydantic_extra__", None) or {})
            # base_url for built-ins (e.g. ollama) is resolved inside Target via
            # the instance overlay / legacy extra_params; per-request base_url
            # (if provided) still wins as an explicit override.

            provider_api_key = plan.provider_api_key

            # Thinking control now lives in each provider (it reads
            # reasoning_effort off the request). The router only forwards the
            # standard intent + the chat_template_kwargs escape hatch.
            #
            # Deprecation shim: the legacy `think` field (ollama-flavored) is
            # retired; translate its one documented intent (think:false = disable
            # reasoning) to the standard vocab, but only if the caller did not
            # already set reasoning_effort. See CONTEXT.md.
            reasoning_effort = request_input.reasoning_effort
            if reasoning_effort is None and request_input.think is not None:
                logger.info("'think' field is deprecated; use reasoning_effort instead")
                if not request_input.think:
                    reasoning_effort = "none"

            if request_input.stream:
                from uniinfer.proxy_services.completion_dispatch import (
                    guarded_stream_sse,
                )

                _stream_gen = guarded_stream_sse(
                    plan,
                    messages=messages_dict,
                    temperature=request_input.temperature,
                    max_tokens=request_input.get_effective_max_tokens(),
                    tools=request_input.tools,
                    tool_choice=request_input.tool_choice,
                    request_id=getattr(request.state, "request_id", None),
                    reasoning_effort=reasoning_effort,
                    chat_template_kwargs=request_input.chat_template_kwargs,
                    extra=extras,
                    usage_note=lambda: get_key_usage().note(
                        api_bearer_token, provider_model, provider_api_key
                    ),
                )
                # Prime the stream: an open-time upstream 429 (RateLimitError)
                # surfaces here as a real HTTP 429 (transparent rate-limit
                # transport) instead of a 200 SSE with a body error chunk.
                # Auto-routed: a 429 here parks the model and falls through to
                # the next candidate instead of failing the request.
                try:
                    _first = await _stream_gen.__anext__()
                except RateLimitError as e:
                    # Fallback to the next chain candidate already happened in
                    # the dispatch core (guarded_stream_sse); reaching this
                    # handler means the chain is exhausted or not auto-routed.
                    _detail = (
                        f"{e} | Response: {e.response_body}"
                        if e.response_body
                        else str(e)
                    )
                    get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
                    get_stats().record(
                        provider_model,
                        status=429,
                        latency_ms=(time.monotonic() - _req_t0) * 1000,
                        usage=None,
                    )
                    _headers = (
                        {"Retry-After": str(int(e.retry_after))}
                        if e.retry_after
                        else {}
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "message": _detail,
                                "type": "rate_limit",
                                "code": 429,
                                **(
                                    {"retry_after": e.retry_after}
                                    if e.retry_after
                                    else {}
                                ),
                            },
                            "model": provider_model,
                        },
                        headers=_headers,
                    )
                # Dead-upstream fallback on prime also lives in the core now
                # (guarded_stream_sse); an exhausted chain re-raises and lands
                # in the outer error mapping below.

                async def _relay() -> AsyncGenerator[str, None]:
                    yield _first  # the primed (welcome) chunk
                    async for _item in _stream_gen:
                        yield _item

                _stream_headers = {
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                }
                if getattr(request.state, "routed_model", None):
                    _stream_headers["x-routed-model"] = str(request.state.routed_model)
                return StreamingResponse(
                    _relay(),
                    media_type="text/event-stream",
                    headers=_stream_headers,
                )

            # Non-stream dispatch (empty-stream guard + 429/dead-upstream
            # fallback) lives in the shared core.
            from uniinfer.proxy_services.completion_dispatch import guarded_complete

            full_content = await guarded_complete(
                plan,
                messages=messages_dict,
                temperature=request_input.temperature,
                max_tokens=request_input.get_effective_max_tokens(),
                tools=request_input.tools,
                tool_choice=request_input.tool_choice,
                reasoning_effort=reasoning_effort,
                chat_template_kwargs=request_input.chat_template_kwargs,
                extra=extras,
            )

            raw_content = full_content.message.content
            tool_calls = full_content.message.tool_calls
            thinking = getattr(full_content, "thinking", None)
            content = normalize_nonstream_content(raw_content, tool_calls)
            finish_reason = getattr(full_content, "finish_reason", None) or (
                "tool_calls" if tool_calls else "stop"
            )

            message_obj = ChatMessageOutput(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                reasoning_details=getattr(full_content, "reasoning_details", None),
            )
            if thinking and not is_openai_strict_mode():
                message_obj.reasoning_content = thinking

            usage = None
            raw_usage = getattr(full_content, "usage", None)
            if raw_usage and isinstance(raw_usage, dict):
                usage = raw_usage
            elif raw_usage:
                usage = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(raw_usage, "completion_tokens", 0)
                    or 0,
                    "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
                }

            response_data = NonStreamingChatCompletion(
                model=provider_model,
                choices=[
                    NonStreamingChoice(message=message_obj, finish_reason=finish_reason)
                ],
                usage=usage,
            )
            _routed_headers = (
                {"x-routed-model": str(getattr(request.state, "routed_model", ""))}
                if getattr(request.state, "routed_model", None)
                else None
            )
            if is_openai_strict_mode():
                get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
                get_stats().record(
                    provider_model,
                    status=200,
                    latency_ms=(time.monotonic() - _req_t0) * 1000,
                    usage=usage,
                )
                return JSONResponse(
                    content=response_data.model_dump(exclude_none=True),
                    headers=_routed_headers,
                )
            get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
            get_stats().record(
                provider_model,
                status=200,
                latency_ms=(time.monotonic() - _req_t0) * 1000,
                usage=usage,
            )
            return JSONResponse(
                content=response_data.model_dump(), headers=_routed_headers
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(
                status_code=getattr(e, "status_code", 401) or 401, detail=detail
            )
        except RateLimitError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            _ra = getattr(e, "retry_after", None)
            raise HTTPException(
                status_code=getattr(e, "status_code", 429) or 429,
                detail=detail,
                headers={"Retry-After": str(max(1, int(_ra)))} if _ra else None,
            )
        except HTTPException:
            raise
        except ProviderError as e:
            detail = f"Provider Error ({provider_name}): {e}"
            if e.response_body:
                detail = f"{detail} | Response: {e.response_body}"
            sc = getattr(e, "status_code", 500) or 500
            get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
            get_stats().record(
                provider_model,
                status=sc,
                latency_ms=(time.monotonic() - _req_t0) * 1000,
                usage=None,
            )
            raise HTTPException(status_code=sc, detail=detail)
        except UniInferError as e:
            raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
        except Exception as e:
            logger.exception(
                "Unexpected error in /v1/chat/completions: %s: %s", type(e).__name__, e
            )
            get_key_usage().note(api_bearer_token, provider_model, provider_api_key)
            get_stats().record(
                provider_model,
                status=500,
                latency_ms=(time.monotonic() - _req_t0) * 1000,
                usage=None,
            )
            raise HTTPException(
                status_code=500, detail=f"Internal Server Error: {type(e).__name__}"
            )
        finally:
            # Release the per-request provider + its httpx client. A Target owns
            # a provider that opens an httpx.AsyncClient; without this every
            # request leaks a connection pool. The streaming path is exempt —
            # the generator closes the target itself once the stream ends.
            if target is not None and not request_input.stream:
                try:
                    await target.aclose()
                except Exception:
                    logger.debug("target.aclose() failed", exc_info=True)

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(
        request: Request,
        request_input: EmbeddingRequest,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        provider_model = request_input.model
        input_texts = request_input.input

        try:
            provider_name, _ = parse_provider_model(provider_model)
            if not instance_requires_api_key(provider_name):
                # Keyless/key-optional endpoint (local ollama, vLLM, LM Studio):
                # best-effort fetch its own credgoo key if one exists (e.g. a
                # bearer-protected ollama), else proceed with None. base_url is
                # resolved inside get_embeddings via the instance overlay.
                from credgoo import get_api_key as _get_credgoo_key

                try:
                    provider_api_key = _get_credgoo_key(
                        _resolve_credgoo_service(provider_name)
                    )
                except Exception:
                    provider_api_key = None
                base_url = None
            else:
                provider_api_key = verify_provider_access(
                    api_bearer_token, provider_name
                )
                base_url = None

            embeddings_result = await run_in_threadpool(
                get_embeddings,
                input_texts=input_texts,
                provider_model_string=provider_model,
                provider_api_key=provider_api_key,
                base_url=base_url,
            )
            get_key_usage().note(api_bearer_token, provider_model, provider_api_key)

            embedding_data = [
                EmbeddingData(embedding=embedding, index=i)
                for i, embedding in enumerate(embeddings_result["embeddings"])
            ]
            response_data = EmbeddingResponse(
                data=embedding_data,
                model=provider_model,
                usage=embeddings_result["usage"],
            )
            return JSONResponse(content=response_data.model_dump())

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(
                status_code=getattr(e, "status_code", 401) or 401, detail=detail
            )
        except RateLimitError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            _ra = getattr(e, "retry_after", None)
            raise HTTPException(
                status_code=getattr(e, "status_code", 429) or 429,
                detail=detail,
                headers={"Retry-After": str(max(1, int(_ra)))} if _ra else None,
            )
        except ProviderError as e:
            detail = f"Provider Error ({provider_name}): {e}"
            if e.response_body:
                detail = f"{detail} | Response: {e.response_body}"
            raise HTTPException(
                status_code=getattr(e, "status_code", 500) or 500, detail=detail
            )
        except UniInferError as e:
            raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error in /v1/embeddings: %s: %s", type(e).__name__, e
            )
            raise HTTPException(
                status_code=500, detail=f"Internal Server Error: {type(e).__name__}"
            )

    return router
