# Auto-Routing: `auto@free`

Virtual model ids that resolve to a concrete `provider@model` at request time.
v0 ships `auto@free` (chat completions only).

> **v1 note:** the per-model usability index (`GET /v1/health/models` —
> Apdex TTFT zones × Laplace-smoothed error rate × traffic confidence) now
> provides the data layer for the planned v1 stats-based ordering. Wiring it
> into the router is a separate, deliberate decision after the numbers have
> baked against real traffic.

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
concrete `provider@model` that served the request — der Header steht auf BEIDEN
Pfaden (non-stream JSONResponse + SSE StreamingResponse); im SSE-Fall ist der
Resolve zusätzlich in jedem Chunk (`chunk.model`). Every routing decision is
logged: `[auto-route] auto@free -> provA@model (reason; N candidates, skipped: …)`.

### Routing-Preview (trocken, ohne Call)

`GET /v1/routing/auto` beantwortet „Welches Modell würde auto@free JETZT
bedienen?" ohne Upstream-Call und ohne Token-Kosten:

```bash
curl http://localhost:<port>/v1/routing/auto
# {"object": "routing", "model": "auto@free",
#  "resolved": "mistral@codestral-2508", "reason": "weight",
#  "candidates": 259,
#  "skipped": {"excluded": 0, "degraded": 0, "quota": 0, "parked": 2},
#  "fallbacks": ["mistral@codestral-latest", "..."],
#  "parked": {"kilo@step-5-preview-free": 812.3},   # Rest-Cooldown-Sekunden
#  "dead_parks": {"kilo@step-5-preview-free": 2}}   # Streak bis zur Auto-Exclusion
```

`resolved: null` = der Router hätte gerade nichts verfügbar (ein echter
Request würde 503 geben) — die Antwort bleibt 200 und zeigt im
`skipped`-Breakdown, WARUM jeder Kandidat rausgefallen ist. Das ist der
Health-Check für „funktioniert auto@free gerade sauber?"; `?model=` nimmt
einen anderen virtuellen Selektor, Nicht-Auto-Modelle → 400.

## How a model is chosen (v0 — deterministic, no scoring)

Candidate set: catalog models with `access: "free"`, `status: "active"`,
`type: "chat"` (`models.json` via `Catalog().list_resolved()`).

Filters, in order:

1. **Operator exclude** — `exclude: true` override drops the model.
2. **Degraded** — models flagged by the ReliabilityMonitor are skipped.
3. **Provider quota pressure** — a provider whose live utilization (7d daily
   average vs its documented `provider_limits.json` free-tier request limit)
   is ≥ 80% is skipped entirely.

Ordering of what survives: operator `pin` first, then personal `prefer`
(per caller, see below), then operator `weight` (default 1.0, higher first),
then catalog order as the stable tiebreak.

## Personal preferences (per caller)

`routing_preferences.json` (default `~/.config/uniinfer/routing_preferences.json`,
override via `UNIINFER_ROUTING_PREFS_FILE`) — mtime-hot-reloaded, edit takes
effect on the next request, no restart. Keyed by token NAME (stable,
human-editable): the request bearer's sha256 maps to a name via the auth-token
metadata sidecar (`auth_tokens.meta.json` next to the allowlist). Anonymous
callers and unknown tokens get the `default` section.

```json
{
  "version": 1,
  "default": {
    "prefer": { "providers": [], "models": [] },
    "avoid":  { "providers": ["kilo"], "models": [] }
  },
  "tokens": {
    "johann-cli":  { "avoid":  { "providers": ["mistral"] } },
    "klark0-web":  { "prefer": { "models": ["mistral@codestral-2508"] } }
  }
}
```

Semantics:

- **`avoid` = harter persönlicher Filter** — eigener Skip-Grund `avoided` im
  Breakdown (Logs, 503-Pfad, Preview); gilt für die ganze Kette, Fallbacks
  routen nie in etwas Vermiedenes. Alles vermieden → 503 mit `avoided: N`.
- **`prefer` = Boost-Tier zwischen Operator-Pin und weight** — operator `pin`
  bleibt darüber (Zuverlässigkeits-Entscheidung schlägt Geschmack).
- Modell-Einträge matchen bare und `provider@model`-Form; Provider
  case-insensitive. Ein model-level `avoid` schlägt ein provider-level
  `prefer` (Filter > Boost).
- Token-Sektionen **addieren** (Union) auf die `default`-Sektion — entfernen
  kann man einen Default-Eintrag nur per Datei-Edit, nicht per Token.
- Prefer/avoid-Modellnamen, die kein Katalogmodell matchen, warnen einmal
  pro Datei-Revision ins Log (Typo-Schutz).
- **Direkte (non-auto) Calls sind unberührt** — Preferences steuern nur das
  Auto-Routing (gleiche Philosophie wie `routing_exclude`).

`GET /v1/routing/auto` mit `Authorization`-Header zeigt DIESE Caller-Sicht:
`preferences: {caller: "johann-cli", active: true, avoided: 14}`.

## Exploration (optional): rotate the free pool, cache-friendly

Default off. `UNIINFER_AUTO_EXPLORE` in the proxy env:

| value | behavior |
|---|---|
| `off` (default) | deterministic v0 order (pin > prefer > weight > catalog) |
| `on` / `least-served` | plain tier ordered by fewest observed requests (7d), random tiebreak |
| `random` | uniform random pick WITHIN the pool zone (the K least-served — not the whole catalog) |

Knobs:
- `UNIINFER_AUTO_EXPLORE_POOL` (default 50) — the exploration zone = the K
  least-observed candidates. Heavily-served models never re-enter the draw.
- `UNIINFER_AUTO_EXPLORE_HOLD_S` (default 900) — **the cache knob**: the
  exploration target is HELD for this window; consecutive requests reuse it
  so provider prompt caches stay warm (`cached_tokens` climbs instead of
  thrashing) and the health index gets coherent per-model samples. `0` =
  per-request rotation (cache-hostile, discouraged).

Why smart instead of naive-random: per-request uniform rotation over ~250
free models means every request starts with a cold prompt cache AND deep
catalog junk gets the same weight as good models. Least-served-first gives
full coverage over time (every model eventually observed → the health index
fills); pins and personal `prefer` still outrank exploration; `avoid` still
filters. The arc: explore → measure → exploit (v1 score ordering replaces
least-served naturally — well-tested models lose the exploration bonus).

Exploration safety: stream-only upstreams answering 400 ("stream:true is
required") classify as dead-upstream — the request falls through the chain,
the model parks (1h) and auto-excludes from routing after 3 parks. Direct
streaming calls to it keep working.

Read-only views never mutate the hold: `GET /v1/routing/auto` follows (but
doesn't set) it and reports `explore: {mode, pool, held_s}`; the fallback-
chain construction likewise. The `[auto-route]` journal line shows
`reason=explore`.

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
