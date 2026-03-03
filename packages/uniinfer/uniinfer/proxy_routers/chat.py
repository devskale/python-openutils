import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from uniinfer.auth import get_optional_proxy_token, verify_provider_access
from uniinfer.errors import AuthenticationError, ProviderError, RateLimitError, UniInferError
from uniinfer.proxy_schemas.chat import (
    ChatCompletionRequestInput,
    ChatMessageOutput,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    NonStreamingChatCompletion,
    NonStreamingChoice,
)
from uniinfer.proxy_services.streaming import astream_response_generator, is_openai_strict_mode, normalize_nonstream_content
from uniinfer.uniioai import aget_completion, astream_completion, get_embeddings

logger = logging.getLogger("uniioai_proxy")


def create_chat_router(
    *,
    parse_provider_model: Callable[..., tuple[str, str]],
    provider_configs: dict[str, Any],
    limiter: Any,
    get_chat_rate_limit: Callable[[], str],
    get_embeddings_rate_limit: Callable[[], str],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    @limiter.limit(get_chat_rate_limit)
    async def chat_completions(
        request: Request,
        request_input: ChatCompletionRequestInput,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        base_url = request_input.base_url
        provider_model = request_input.model
        messages_dict = [msg.model_dump() for msg in request_input.messages]

        try:
            provider_name, _ = parse_provider_model(provider_model)
            if provider_name == "ollama" and base_url is None:
                base_url = provider_configs.get("ollama", {}).get("extra_params", {}).get("base_url")

            provider_api_key = verify_provider_access(api_bearer_token, provider_name)

            if request_input.stream:
                return StreamingResponse(
                    astream_response_generator(
                        astream_completion=astream_completion,
                        messages=messages_dict,
                        provider_model=provider_model,
                        temp=request_input.temperature,
                        max_tok=request_input.get_effective_max_tokens(),
                        provider_api_key=provider_api_key,
                        base_url=base_url,
                        tools=request_input.tools,
                        tool_choice=request_input.tool_choice,
                        request_id=getattr(request.state, "request_id", None),
                        reasoning_effort=request_input.reasoning_effort,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            full_content = await aget_completion(
                messages=messages_dict,
                provider_model_string=provider_model,
                temperature=request_input.temperature,
                max_tokens=request_input.get_effective_max_tokens(),
                provider_api_key=provider_api_key,
                base_url=base_url,
                tools=request_input.tools,
                tool_choice=request_input.tool_choice,
                reasoning_effort=request_input.reasoning_effort,
            )

            raw_content = full_content.message.content
            tool_calls = full_content.message.tool_calls
            thinking = getattr(full_content, "thinking", None)
            content = normalize_nonstream_content(raw_content, tool_calls)
            finish_reason = getattr(full_content, "finish_reason", None) or ("tool_calls" if tool_calls else "stop")

            message_obj = ChatMessageOutput(role="assistant", content=content, tool_calls=tool_calls)
            if thinking and not is_openai_strict_mode():
                message_obj.reasoning_content = thinking

            response_data = NonStreamingChatCompletion(
                model=provider_model,
                choices=[NonStreamingChoice(message=message_obj, finish_reason=finish_reason)],
            )
            if is_openai_strict_mode():
                return JSONResponse(content=response_data.model_dump(exclude_none=True))
            return JSONResponse(content=response_data.model_dump())

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(status_code=getattr(e, "status_code", 401) or 401, detail=detail)
        except RateLimitError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(status_code=getattr(e, "status_code", 429) or 429, detail=detail)
        except HTTPException:
            raise
        except ProviderError as e:
            detail = f"Provider Error ({provider_name}): {e}"
            if e.response_body:
                detail = f"{detail} | Response: {e.response_body}"
            raise HTTPException(status_code=getattr(e, "status_code", 500) or 500, detail=detail)
        except UniInferError as e:
            raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
        except Exception as e:
            logger.exception("Unexpected error in /v1/chat/completions: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {type(e).__name__}")

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    @limiter.limit(get_embeddings_rate_limit)
    async def create_embeddings(
        request: Request,
        request_input: EmbeddingRequest,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        provider_model = request_input.model
        input_texts = request_input.input

        try:
            provider_name, _ = parse_provider_model(provider_model)
            if provider_name == "ollama":
                provider_api_key = None
                base_url = provider_configs.get("ollama", {}).get("extra_params", {}).get("base_url")
            else:
                provider_api_key = verify_provider_access(api_bearer_token, provider_name)
                base_url = None

            embeddings_result = await run_in_threadpool(
                get_embeddings,
                input_texts=input_texts,
                provider_model_string=provider_model,
                provider_api_key=provider_api_key,
                base_url=base_url,
            )

            embedding_data = [EmbeddingData(embedding=embedding, index=i) for i, embedding in enumerate(embeddings_result["embeddings"])]
            response_data = EmbeddingResponse(data=embedding_data, model=provider_model, usage=embeddings_result["usage"])
            return JSONResponse(content=response_data.model_dump())

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AuthenticationError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(status_code=getattr(e, "status_code", 401) or 401, detail=detail)
        except RateLimitError as e:
            detail = f"{e} | Response: {e.response_body}" if e.response_body else str(e)
            raise HTTPException(status_code=getattr(e, "status_code", 429) or 429, detail=detail)
        except ProviderError as e:
            detail = f"Provider Error ({provider_name}): {e}"
            if e.response_body:
                detail = f"{detail} | Response: {e.response_body}"
            raise HTTPException(status_code=getattr(e, "status_code", 500) or 500, detail=detail)
        except UniInferError as e:
            raise HTTPException(status_code=500, detail=f"UniInfer Error: {e}")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in /v1/embeddings: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {type(e).__name__}")

    return router
