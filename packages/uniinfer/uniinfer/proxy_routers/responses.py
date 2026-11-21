"""Responses API surface (``POST /v1/responses``) — thin adapter over the
shared completion dispatch + the responses translation core.

uniinfer IS the normalization layer: every chat provider speaks Responses
here (translated), native-Responses backends are a follow-up fast-path. v1 is
stateless (see responses_translate.reject_unsupported).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uniinfer.auth import get_optional_proxy_token
from uniinfer.proxy_services.completion_dispatch import (
    error_to_http_exception,
    guarded_complete,
    guarded_stream_events,
    make_meter,
    normalize_roles,
    resolve_dispatch,
)
from uniinfer.proxy_services.key_usage import get_key_usage
from uniinfer.responses_translate import (
    ResponsesEventEmitter,
    completion_to_response,
    reject_unsupported,
    request_to_chat,
)

logger = logging.getLogger("uniioai_proxy")


def create_responses_router(parse_provider_model) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    async def create_response(
        request: Request,
        api_bearer_token: str | None = Depends(get_optional_proxy_token),
    ):
        import time

        body = await request.json()
        provider_model = str(body.get("model") or "")
        request.state.provider_model = provider_model
        _t0 = time.monotonic()
        plan = None

        try:
            unsupported = reject_unsupported(body)
            if unsupported:
                raise HTTPException(status_code=400, detail=unsupported)
            chat = request_to_chat(body)

            plan = resolve_dispatch(
                provider_model,
                api_bearer_token,
                own_key=request.headers.get("x-provider-key"),
            )
            request.state.provider_model = plan.provider_model
            if plan.routed_model:
                request.state.routed_model = plan.routed_model
            meter = make_meter(api_bearer_token, plan.provider_model, plan.provider_api_key, _t0)
            messages = normalize_roles(chat["messages"], plan.provider_name)

            if body.get("stream"):
                emitter = ResponsesEventEmitter(plan.provider_model)

                async def _relay():
                    try:
                        gen = guarded_stream_events(
                            plan,
                            messages=messages,
                            temperature=chat["temperature"],
                            max_tokens=chat["max_tokens"],
                            tools=chat["tools"],
                            tool_choice=chat["tool_choice"],
                            request_id=getattr(request.state, "request_id", None),
                            usage_note=lambda: get_key_usage().note(
                                api_bearer_token, plan.provider_model, plan.provider_api_key
                            ),
                        )
                        # Prime before committing the SSE 200 (transparent 429).
                        first = await gen.__anext__()
                        for event in emitter.start():
                            yield event
                        for event in emitter.feed(first):
                            yield event
                        async for chunk in gen:
                            for event in emitter.feed(chunk):
                                yield event
                        for event in emitter.finish():
                            yield event
                    except Exception as e:
                        logger.warning("responses stream failed mid-flight: %s", e)
                        yield emitter.failed(e)
                    finally:
                        try:
                            await plan.target.aclose()
                        except Exception:
                            logger.debug("target.aclose() failed after responses stream", exc_info=True)

                return StreamingResponse(
                    _relay(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            full_content = await guarded_complete(
                plan,
                messages=messages,
                temperature=chat["temperature"],
                max_tokens=chat["max_tokens"],
                tools=chat["tools"],
                tool_choice=chat["tool_choice"],
            )
            response_obj = completion_to_response(full_content, plan.provider_model)
            meter(200, None, response_obj.get("usage"))
            headers = (
                {"x-routed-model": str(plan.routed_model)} if plan.routed_model else None
            )
            return JSONResponse(content=response_obj, headers=headers)
        except HTTPException:
            raise
        except Exception as e:
            raise error_to_http_exception(
                e,
                plan.provider_name if plan is not None else provider_model.split("@", 1)[0],
                meter=None,
            )
        finally:
            if plan is not None and not body.get("stream"):
                try:
                    await plan.target.aclose()
                except Exception:
                    logger.debug("target.aclose() failed after responses complete", exc_info=True)

    return router
