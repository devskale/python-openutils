# Memory & stability decisions (uniioai-proxy)

Definitive record of the 2026-08 memory/stability investigation and fix on the
`amd` uniioai-proxy (0.8.3 → 0.8.40). Every decision is grounded in measurement
+ web research.

## TL;DR

- **Root cause found and fixed**: httpcore #1093 — HTTP/1.1 connection-pool
  slots orphaned under request cancellation; the pool degraded monotonically
  to wedge. **Fix: HTTP/2 on all clients** (`http2=True`, `h2>=4.1.0`).
- **Allocator: plain glibc** (jemalloc evaluated, dropped — neutral).
- **Rate limiting: transparent 429** (adaptive limiter + slowapi both removed).
- **Middleware: pure-ASGI** (BaseHTTPMiddleware replaced — it leaked under SSE).
- **Monitoring: `/health`** + opt-in tracer (off by default, zero overhead).

---

## 1. The root cause — httpcore #1093 (HTTP/1.1 pool leak)

The proxy **wedged** (D-state, `/health` timeout) after ~12h of sustained
chat/image load, RSS climbing monotonically to the 200M cgroup cap.

**Tracemalloc pinpointed it**: `json/decoder.py:361` + `starlette/requests.py:259`
+ `httpx/_content.py:179` grew every 5-min interval; `tuple` count climbed
monotonically (24k→42k→65k→74k, never dropped). The retained growth was
per-request objects (pydantic `model_dump`, httpx request-body encode) held by
**orphaned connection-pool slots**.

**Grounded in** [encode/httpcore#1093](https://github.com/encode/httpcore/issues/1093)
(opened Jul 2026, httpcore 1.0.9 — our exact version): *"cancelled requests
under pool contention permanently leak connection slots… only recreating the
pool (process restart) recovers… HTTP/1.1-specific; HTTP/2 multiplexes, rarely
reaches contention."*

Our SSE path **cancels pooled requests constantly** (streaming heartbeat
timeouts via `asyncio.wait_for` + client aborts on 429) under concurrent load —
the exact trigger.

## 2. The fix — HTTP/2 standard (0.8.26)

`http2=True` on all 6 `httpx.AsyncClient` sites + `h2>=4.1.0` dep. HTTP/2
multiplexes all streams over **one connection per host** → no pool contention
→ no orphan slots. ALPN falls back to HTTP/1.1 for hosts that don't speak h2
(1 of 18: internlm). Verified: 17/18 upstreams negotiate h2.

**Post-fix**: `tuple` count plateaus (~48k, fluctuates with load, drops
between bursts); `objs` stable; RSS bounded (105–145MB under stress vs
monotonic climb before). Confirmed over sustained stress.

## 3. Allocator stack — glibc (0.8.13–0.8.18)

**Evaluated**: jemalloc (`LD_PRELOAD`) vs glibc on amd. jemalloc governs only
~25% of RSS (`live_allocated` ~24MB of ~96MB); the other ~70MB is CPython
`pymalloc` arenas that bypass malloc entirely. jemalloc was neutral-to-slightly-
worse on the malloc'd slice. `PYTHONMALLOC=malloc` tested + reverted (no benefit).

**Final**: plain glibc with `MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=65536`,
`MALLOC_TRIM_THRESHOLD_=65536`. No jemalloc.

## 4. Rate limiting — transparent 429 (0.8.14–0.8.15, 0.8.28, 0.8.38)

- **TU adaptive throttle removed** (0.8.15): upstream 429s are relayed to the
  caller as `RateLimitError` immediately (was: up-to-120s internal backoff-retry
  that stalled streams). For streaming, the SSE is primed before committing so
  an open-time 429 is a real `HTTP 429`.
- **slowapi removed** (0.8.28): its `SlowAPIASGIMiddleware` re-sent
  `http.response.start` on every body chunk, corrupting multi-chunk responses
  (the webdemo truncated at 64KB). It also never fired (storage stayed empty;
  real rate limiting is upstream TU + bearer auth).
- **`ratelimit.py` deleted** (0.8.38): the `AdaptiveRateLimiter` (395 LOC) was
  vestigial infrastructure — no provider registered a limiter after TU's removal.

## 5. Middleware — pure-ASGI (0.8.21)

`BaseHTTPMiddleware` (2 `@app.middleware("http")` + `SlowAPIMiddleware`) leaked
under streaming/SSE (Starlette #1012 — buffers every response through an anyio
memory stream). Replaced with **pure-ASGI** equivalents:
`LeanHTTPMiddleware` (request-id logging + body-size limit, streams transparently)
+ CORS (already pure-ASGI).

## 6. Dispatch modules (0.8.30–0.8.34)

Three deep modules following the `Target` (chat) pattern:
- **`ImageTarget`** (`uniinfer/images.py`) — 3 dialect adapters (generic
  OpenAI-compatible, cloudflare, pollinations) behind one interface.
- **`TTSTarget` / `STTTarget`** (`uniinfer/audio.py`) — lazy provider registry.

## 7. Monitoring — `/health` + opt-in tracer

- **`/health`** (always on): `status` (ok/warn/crit), RSS/peak/swap, cgroup
  caps + headroom, page-fault rate, event-loop latency, version, uptime.
  Env-configurable caps (`UNIINFER_MEM_HIGH_MB`/`UNIINFER_MEM_MAX_MB`) for
  non-cgroup hosts.
- **Tracer** (`UNIINFER_MEM_TRACE=1`, off by default): logs RSS + gc object
  census + tracemalloc allocation-site diffs to `logs/mem_trace.log`.
  `UNIINFER_MEM_TRACE_MALLOC=1` enables tracemalloc (expensive on large
  responses — do NOT leave on in production).
