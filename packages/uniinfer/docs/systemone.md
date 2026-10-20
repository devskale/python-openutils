# Systemone — eine Tür für System-One-Modelle (decision + extraction)

`POST /v1/systemone` bedient Modelle, die strukturierte Reads statt freien Text
liefern. Kanonische Implementierung + ausführlicher Vertrag:
[`uniinfer/proxy_routers/systemone.py`](../uniinfer/proxy_routers/systemone.py)
(Docstring ist Source of Truth — dieses Doc hält Konzept, Konfiguration und
Beispiele, kein Duplikat der Logik).

## Die zwei Dialekte

| Dialekt | Payload | Antwort | Modelle |
|---|---|---|---|
| **decision** | `{model, state, questions}` | `{answers}` — choice/noul/score + Wahrscheinlichkeiten | jev, clef-Familie, d3-mini (dgx) |
| **extraction** | `{model, state, fields}` | typisierte Feldwerte + Confidence + Verteilungsstatistik | clairic (CF) |

Dialekt-Erkennung: **Katalog-Typ ist Autorität** (`type_overrides.json`,
denselbe Typ zeigt `/v1/models`), Payload-Shape (`questions` vs `fields`) ist
Fallback für ungetypte Modelle. Konflikt → 400. Neue Modelle: Payload-Shape
reicht, Typisierung per `type_overrides.json` nachziehen.

## Die zwei Transporte (Instanz-Eigenschaft, nicht Modell- oder Provider-Code)

| Instanz | Weg | Dialekte |
|---|---|---|
| mit `base_url` | **nativer 1:1-Forward** an `{base_url}{systemone_path}` | beide |
| ohne `base_url` (built-in) | **chat-wrap**: Payload als letzte user-message durch den Chat-Pfad | nur decision |

Extraction auf chat-wrap → 400 (Extraction-Modelle erzeugen keine Output-
Tokens, chat-wrap ist physikalisch unmöglich) — nicht 502.

## `systemone_path` pro Instanz

| Wert | URL | Nutzung |
|---|---|---|
| `null` (Default) | `{base}/systemone` | native vLLM/jev-Endpoints (d3mini, dgemma) |
| `""` | `{base}` AS-IS | Instanz-pro-Modell mit vollständiger Run-URL (clef, clef-flash, clef-omni) |
| `"/{model}"` | `{base}/{bare_model}` | **Instanz-pro-Endpoint** für Familien: `cfrun` bedient JEDES CF-System-One-Modell |

## Beispiele

**Decision (d3-mini auf dgx):**

```bash
curl -X POST .../v1/systemone -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{
    "model": "d3mini@vllm-sr/d3-mini",
    "state": "Meine Karte wurde zweimal belastet.",
    "questions": {"team": {"type": "choice", "instructions": "Welches Team?",
                           "criteria": {"billing": "Zahlungen", "versand": "Lieferungen"}}}}'
```

**Extraction (clairic über `cfrun`):**

```bash
curl -X POST .../v1/systemone -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{
    "model": "cfrun@@cf/cloudflare/clairic",
    "state": "Support-Ticket: Paket 88123 seit 5 Tagen weg …",
    "fields": {"team": {"type": "text", "instructions": "Eines von: billing, …"},
               "urgency": {"type": "estimate", "min": 1, "max": 5}}}'
```

Antwort: flacher `{answers, usage}`-Shape — CF `{success, result}`-Wrapper
wird vom Router geflattet.

## Cloudflare-Clairic-Schema (live verifiziert 2026-10-20)

- `state` = **plain string** (kein Objekt)
- Feld-Typen: `text, number, date, time, datetime, duration, list, estimate, forecast`
  — kein `enum`/`boolean`/`compute`
- `estimate` braucht numerische `min`/`max`; `list` nimmt `items` mit skalaren
  Typ-Specs (keine Wert-Listen) — Routing-Auswahl daher via `text` + instructions

## Fehlersemantik

| Situation | HTTP |
|---|---|
| Body ohne `questions` und `fields` | 400 |
| Katalog-Typ ≠ Payload-Dialekt | 400 (nennt den Katalog-Typ) |
| Extraction auf chat-wrap-Instanz | 400 („needs a native instance") |
| Unbekannter Alias | 404 |
| Upstream-Timeout / -Fehler | 504 / 502 |

## Ein Modell/ eine Instanz ergänzen

1. Instanz in `provider_instances.json` (Deployment-Box): `base_url`,
   `provider`, `credgoo_service`, ggf. `systemone_path` — für Familien das
   `"/{model}"`-Muster (siehe `cfrun`).
2. Modell in `type_overrides.json` typen: `"decision"` oder `"extraction"`
   (Keys ohne führendes `@`, Lookup normalisiert `@cf/…`-IDs).
3. Fertig — Router, Dialekt-Check und `/v1/models`-Typierung laufen deklarativ.
