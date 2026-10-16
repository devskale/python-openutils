# Auto-Routing: `auto@free`

Virtual model ids that resolve to a concrete `provider@model` at request time.
v0 ships `auto@free` (chat completions only).

## Usage

```bash
curl -X POST http://localhost:<port>/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "auto@free", "messages": [{"role": "user", "content": "Hi"}]}'
```

The response `model` field and the `x-routed-model` response header carry the
concrete `provider@model` that served the request. Every routing decision is
logged: `[auto-route] auto@free -> provA@model (reason; N candidates, skipped: …)`.

## How a model is chosen (v0 — deterministic, no scoring)

Candidate set: catalog models with `access: "free"`, `status: "active"`,
`type: "chat"` (`models.json` via `Catalog().list_resolved()`).

Filters, in order:

1. **Operator exclude** — `exclude: true` override drops the model.
2. **Degraded** — models flagged by the ReliabilityMonitor are skipped.
3. **Provider quota pressure** — a provider whose live utilization (7d daily
   average vs its documented `provider_limits.json` free-tier request limit)
   is ≥ 80% is skipped entirely.

Ordering of what survives: `pin` first, then operator `weight` (default 1.0,
higher first), then catalog order as the stable tiebreak.

## Operator controls (the existing override seam)

`model_overrides.json` (or `PUT /v1/models/overrides/{model_id}`) gains three
routing fields:

| Field | Type | Effect |
|---|---|---|
| `weight` | float (default 1.0) | preference ordering; higher = earlier |
| `pin` | bool | always route here when available (overrides weight) |
| `exclude` | bool | never route here |

```bash
curl -X PUT .../v1/models/overrides/mistral-small-latest \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"pin": true}'
```

## Known limits (by design, v0)

- **Chat only.** Images/audio/embeddings do not auto-route.
- **No mid-stream rescue.** Streaming requests that die after the first chunk
  cannot re-route; only failures before the first chunk can fall through to
  the next candidate (non-streaming path).
- **No quality signal yet.** v0 orders by operator preference only; live
  stats scoring (error rate, TTFT) is v1, judged quality weights v2 — see
  the `auto-free-router` issue in the kanban.
- **Cold start.** A model with no traffic history has no stats penalty in v0
  (there is no scoring); ordering is deterministic regardless of traffic.
