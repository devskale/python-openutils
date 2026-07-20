# Testsuite Fixtures — Doks & Cases

Selbstständige, **PII-freie** Testdaten für die uniinfer-Testsuite. Strikt getrennt:

- **`docs/`** = Kontext-Dokumente (nur pseudonymisiert / fiktiv — **niemals Echtdaten**).
- **`cases/`** = `doc ↔ query ↔ expected` Paare (ein JSONL pro App/Task).

Die Runner entdecken beides automatisch; ein neuer Testfall = eine Datei + eine JSONL-Zeile, **kein Code**.

```
fixtures/
├─ docs/
│  ├─ ofs_audit/   Preisblätter (Bieter als Musterbieter_A/B/C anonymisiert)
│  └─ realworld/   fiktive Leistungsverzeichnisse (~0,3k–17k Token, context-scaling)
└─ cases/
   ├─ ofs_audit.jsonl          Preisblatt-Audit (29)
   ├─ strukt2meta_meta.jsonl   Metadaten-Extraktion (3)
   ├─ strukt2meta_bdok.jsonl   bdok/adok-Klassifikation + Kriterium-Check (4, think-sensitiv)
   ├─ agentos_rag.jsonl        RAG-Anwort / Retriever (3, no-think)
   ├─ pdf2md_cleanup.jsonl     PDF-Wirrwarr → sauberes Markdown (2)
   └─ realworld_lv.jsonl       LV-Zusammenfassung über Kontext-Spanne (4)
```

## Case-Schema (eine Zeile pro Case in `cases/*.jsonl`)

```json
{
  "id": "lv800_summary",
  "app": "realworld",
  "task": "lv_summary",
  "query": "Fasse dieses Leistungsverzeichnis zusammen …",
  "docs": ["realworld/lv_reinigung_800pos.md"],
  "expected": {
    "kind": "set",
    "answer_contains": ["483.137,72", "800"],
    "must_not_contain": [],
    "fields": {},
    "citations": []
  },
  "difficulty": "hard",
  "tags": ["lv", "context-scaling"],
  "source": "strukt2meta/prompts/…"
}
```

`docs` sind Pfade **relativ zu `docs/`**. Mehrere Docs werden konkateniert (ein Kontext).

### `expected.kind` — Ground-Truth passend zum Task-Typ

| kind | wann | geprüft via |
|---|---|---|
| `exact` | einzelner Wert (Klassifikation, Routing) | exakt / Float-Toleranz |
| `set` | Extraktion, mehrere akzeptable Formen | Teilstring ODER Float (`answer_contains`) |
| `rubric` | Freitext (RAG, Zusammenfassung) | `answer_contains` ∧ `must_not_contain` |
| `reference` | Paraphrase (Markdown-Cleanup) | semantische Ähnlichkeit zu Referenz |

> **Bewertung passiert extern** (Mensch oder LLM-as-Judge) — nicht im Runner. Der Runner schreibt nur `answer` + Metriken ins JSONL.

## Erweitern (kein Code)

1. **Neuer Doc:** Datei nach `docs/<app>/` (PII-frei! Namen/Adressen/UID/IBAN fiktiv).
2. **Neuer Case:** eine Zeile an `cases/<app>.jsonl` anhängen (Schema siehe oben) — oder neues `cases/<neueapp>.jsonl`.
3. Fertig — `qa_gen.py` findet alles via Glob.

## Runner

```bash
# alle Modelle (models.json) × alle Cases × think/no-think
uv run python testsuite/qa_gen.py --config testsuite/models.json --reasonings none,high

# nur eine App / ein Case-File
uv run python testsuite/qa_gen.py --app strukt2meta
uv run python testsuite/qa_gen.py --cases fixtures/cases/realworld_lv.jsonl

# tok/s über reale Kontext-Spanne (echte prompt_tokens als x-Achse)
uv run python testsuite/bench_realworld.py --config testsuite/models.json
```

**JSONL pro Lauf** (`runs/*.jsonl`, gitignored): `model_id, case_id, app, task, reasoning,
query, docs, expected, answer, tok_per_s, output_tokens, prompt_tokens, latency_s,
status_code, error`. HTTP 402/429/4xx/5xx werden als `status_code` erfasst (kein Absturz).

## Modelle / Secrets (gitignored)

`testsuite/models.json` (**nicht committet**) listet die Ziele:
```json
{ "models": [
  {"name":"tu qwen-3.6-35b","base_url":"https://amd1.mooo.com:8123/v1","bearer":"$PROXY_KEY","model_id":"tu@qwen-3.6-35b"},
  {"name":"nim (local)","base_url":"http://localhost:8000/v1","bearer":"EMPTY","model_id":"Nvidia/…","enabled":false}
]}
```
- `bearer`: `EMPTY`/"" = kein Auth · `$VAR`/`env:VAR` = aus `.env` · sonst literal. `.env` wird vom Runner geladen.
- `enabled: false` = übersprungen.

## PII-Policy

- **Niemals** reale Vergabe-/Kundendaten committen. Doks hier sind **pseudonymisiert** (Bieter `Musterbieter_*`, fiktive Firmen/Straßen/Personen/UID/IBAN) oder **frei erfunden**.
- `models.json`, `.env`, `runs/` sind in `.gitignore` — Secrets & Ergebnisse bleiben lokal.
