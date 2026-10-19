# Auto-Routing: `auto@free`

Virtual model ids that resolve to a concrete `provider@model` at request time.
v0 ships `auto@free` (chat completions only).

## Usage

`auto@free` ist ein virtuelles Modell — in `/v1/models` gelistet wie ein
konkretes (`provider: auto`), also über jeden OpenAI-kompatiblen Client
auswählbar:

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
- **Mid-stream failures are not rescued.** Failures before the first chunk
  fall through to the next candidate; once a stream is flowing, the request
  stands or dies with its model.
- **No quality signal yet.** v0 orders by operator preference only; live
  stats scoring (error rate, TTFT) is v1, judged quality weights v2 — see
  the `auto-free-router` issue in the kanban.
- **Cold start.** A model with no traffic history has no stats penalty in v0
  (there is no scoring); ordering is deterministic regardless of traffic.

## Deprecation (manuell + auto)

Ein Modell kann aus dem Routing und den aktiven Listen genommen werden, ohne
es zu löschen — mit Grund und Datum, sichtbar unter `/v1/models/deprecated`:

**Manuell:**

```bash
curl -X POST .../v1/models/deprecate/step-5-preview-free \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"reason": "upstream removed the free tier"}'

curl -X POST .../v1/models/undeprecate/step-5-preview-free -H "Authorization: Bearer $TOKEN"
```

**Auto:** antwortet ein geroutetes Modell wiederholt „Model is unavailable" /
404 (dead upstream — das Modell ist upstream *weg*, nicht flaky), wird es
für 1h geparkt; nach 3 solchen Parks (in Folge — ein erfolgreicher Request
setzt den Streak zurück) schreibt der Router automatisch ein
**routing-scoped** `routing_exclude`-Override (`routing_exclude_source:
"auto"` mit dem Upstream-Fehler als Grund). Das ist bewusst KEINE
Katalog-Deprecation: `auto@free` ist ein virtuelles Modell — der Ausschluss
gilt nur fürs Routing, Katalog-Status und direkte Calls bleiben unberührt.
Lösen lässt sich der Ausschluss per Overrides-API (Feld löschen), die
Katalog-Endpoints deprecate/undeprecate bleiben reine Katalog-Operationen.

## Quota cooldown (parking)

Free-tier tokens run out. When an auto-routed model answers 429 (or a
quota-class error), the router:

1. **parks** the model for 15 minutes (upstream `Retry-After` extends the park
   if longer, capped at 6h; it never shortens below the default),
2. **falls through** to the next candidate in the chain (up to 3 fallbacks),
3. re-resolves the chain on every retry, so parked models drop out live.

Parking is in-memory (a restart re-arms on the first real 429). The parked
set is visible in `[auto-route] parking …` log lines and via
`_QuotaCooldown.snapshot()` (surfaced in /health in a follow-up).
