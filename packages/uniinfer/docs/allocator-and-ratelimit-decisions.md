# Allocator & rate-limit decisions (uniioai-proxy)

Operational/config record of the decisions made after the 2026-08 memory and
429-stall investigation on the `amd` uniioai-proxy (0.8.14 → 0.8.16). These are
conclusions grounded in measurement + web research; keep them if you revisit the
proxy's memory story.

## TL;DR

- The proxy runs the **plain glibc allocator** — **no jemalloc**, no
  `PYTHONMALLOC`. Only glibc `mallopt` tuning is set.
- **TU no longer throttles internally** — upstream `429`s are relayed to the
  client transparently instead of being absorbed by an up-to-120s backoff-retry.
- **No restart watchdog / bounded-worker-lifecycle** was added, and none is
  warranted by the current measurements.
- `/health` is kept as the lightweight, permanent guard.

---

## 1. Allocator stack (systemd unit on `amd`)

**Current env on `/etc/systemd/system/uniioai-proxy.service`:**

```
Environment=MALLOC_ARENA_MAX=2
Environment=MALLOC_MMAP_THRESHOLD_=65536
Environment=MALLOC_TRIM_THRESHOLD_=65536
Environment=PYTHONUNBUFFERED=1
```

**Removed:**
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- `MALLOC_CONF=dirty_decay_ms:1000,muzzy_decay_ms:1000`
- (experimental) `PYTHONMALLOC=malloc` — tested, no benefit, reverted.

**Why:** the A/B allocator evaluation (jemalloc vs glibc, both on amd) showed
jemalloc governs only ~25% of RSS (`live_allocated` ~24MB of ~96MB); the other
~70MB is CPython `pymalloc` small-object arenas, which are mmap'd directly and
**bypass `malloc` entirely** — no C-allocator setting touches them. On the slice
jemalloc does control it was neutral-to-slightly-worse, and its 1s
`dirty_decay_ms` introduced a purge-storm failure mode under bursty large
allocations. The glibc `mallopt` vars above are the ones that matter (large
allocations ≥64KB go to mmap → returned to the OS on free; heap trims eagerly).

Backups on amd: `~/uniioai-proxy.service.bak.jemalloc-removed`.

## 2. Rate limiting — removed, 429 passed through

Decision: **the TU adaptive throttle was removed entirely** (0.8.15). No
`_throttle`/`acquire`/`on_429`/`on_success`, no module-level TU limiter.
Requests go out immediately; an upstream `429` is relayed to the caller:

- **non-streaming** → real `HTTP 429` (`RateLimitError` → `JSONResponse 429`).
- **streaming** → the SSE is primed (first upstream chunk pulled) *before* the
  welcome chunk is committed, so an open-time `429` is a real `HTTP 429`, not a
  `200`-with-error-chunk. Mid-stream errors remain body error chunks.

Only short transport-fault retries (network errors) remain — not rate-limit
retries.

**Why:** the `120s` internal backoff-retry (5→10→20→…→120s per request, up to 4×)
was the root of two symptoms at once — the *stalls* the astro/pi agent saw under
fast deepseek + image load exceeding TU's shared ~25/min quota, and a chunk of
the *memory* growth (many pending SSE generators/backing-off tasks held
simultaneously). Relaying the 429 lets the client do its own backoff.

## 3. No restart watchdog (bounded-worker-lifecycle)

Deliberately **not** implemented. Evaluated and rejected as a *workaround* (per
the gunicorn maintainer: `max_requests` is "a temporary workaround for an
application code leaking"; "mitigates memory leaks, doesn't fix them").

**Why it isn't needed:** the object-count-vs-RSS measurement (recorded below)
shows a **flat live-object count** while RSS moves — the classic "leak that
isn't a leak" signature. RSS is transient load memory (buffers/arenas) that
returns after load under the glibc tuning. There is no real leak to recycle
against, and the stall-driven accumulation that did grow RSS was removed by the
429 pass-through (#2). The `/health` endpoint + the 200M/300M cgroup caps remain
as the guard: if slow long-run build-up ever approaches the cap, `/health`
status flips to `warn`/`crit` and a memory-signal recycle can be revisited as the
*fallback* — not the answer.

## 4. `/health` endpoint (kept)

Added (0.8.13) as the lightweight, **unauthenticated, zero-cost** probe: returns
`status` (ok/warn/crit), RSS/peak/swap vs the cgroup caps + headroom, page-fault
rate, event-loop latency, version, uptime. It also exposes jemalloc
live-vs-retained split (`live_allocated_mb`/`resident_mb`/`unreturned_mb`);
without jemalloc loaded those fields degrade to `null` (as designed).

---

## Evidence (measurements on `amd`, 2026-08)

| Check | Result |
|---|---|
| jemalloc A/B (provider-stream TLS under jemalloc) | flat; pools vs fresh identical — not the leak |
| jemalloc vs glibc (small objects / pymalloc) | identical (+0.1MB) → allocator-independent |
| jemalloc vs glibc (large buffers, tight loop) | glibc retained 6.1MB, jemalloc 8.0MB (worse) |
| jemalloc vs glibc (spaced images, realistic) | identical (+9.7MB) |
| object-count-vs-RSS under load (glibc) | objects ~197k→198.5k (flat) while RSS 88→99MB, then back to ~89 |
| object-count-vs-RSS under `PYTHONMALLOC=malloc` | objects flat; RSS 104 peak / 92 residual — no benefit vs glibc |

**Takeaway:** flat object count + self-recovering RSS = **allocator/arena
transients, not a leak**; every allocator knob (jemalloc, glibc, pymalloc) is
moot. The thing that actually accumulated memory ("stall pile-up") is already
eliminated by the transparent-429 relay.
