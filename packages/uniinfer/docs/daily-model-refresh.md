# Daily Model Refresh Service

## Problem

`models.json` is regenerated on-demand when stale (controlled by `REFETCHTIME`, default 24h).
This means:
- No guaranteed daily refresh
- No visibility into model freshness per provider
- Models that disappear from a provider are immediately deleted from history — no grace period
- No way to distinguish "verified today" from "last verified a week ago"

## Solution

### 1. systemd timer (`uniioai-models-refresh.timer`)

Runs `generate_models.py` once per day at 04:00 UTC.

### 2. Freshness tracking in `_model_history.json`

**Before** (flat string):
```json
{ "openai/gpt-4o": "2026-04-19" }
```

**After** (object with first_seen + last_seen):
```json
{ "openai/gpt-4o": { "first_seen": "2026-04-19", "last_seen": "2026-06-03" } }
```

Backward compatible: on first run with new code, old format is auto-migrated.

### 3. Model lifecycle states

| State | Condition | In models.json? |
|-------|-----------|----------------|
| `fresh` | `last_seen` == today | ✅ yes |
| `stale` | `last_seen` < today, but < 90 days ago | ✅ yes (warned) |
| `pruned` | `last_seen` > 90 days ago | ❌ removed from output |

- `fresh`/`stale` models appear in `/v1/models` with a `freshness` field
- `stale` models also get `days_since_seen: N`
- `pruned` models are kept in `_model_history.json` (for analytics) but excluded from `models.json`

### 4. API changes

Every model in `/v1/models` gets:
```json
{
  "id": "openai/gpt-4o",
  "freshness": "fresh",       // "fresh" | "stale"
  "days_since_seen": 0,        // always present, 0 for fresh
  "last_seen": "2026-06-03"    // date of last live verification
}
```

New endpoint `GET /v1/models/stale?days=90` — lists stale/pruned models with details.

### 5. Deployment

```
# Copy to amd (via deploy.sh or manual)
sudo cp uniioai-models-refresh.service /etc/systemd/system/
sudo cp uniioai-models-refresh.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now uniioai-models-refresh.timer

# Verify
sudo systemctl list-timers | grep models
journalctl -u uniioai-models-refresh --since "1 hour ago"
```

## Files changed

| File | Change |
|------|--------|
| `scripts/generate_models.py` | Track `last_seen`, don't delete stale models, emit freshness stats |
| `uniinfer/proxy_services/models_registry.py` | Add `freshness`/`days_since_seen` to model output, add prune logic |
| `uniinfer/proxy_routers/models.py` | New `/v1/models/stale` endpoint |
| `deploy/uniioai-models-refresh.service` | systemd unit (new) |
| `deploy/uniioai-models-refresh.timer` | systemd timer (new) |
