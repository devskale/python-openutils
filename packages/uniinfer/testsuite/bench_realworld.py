#!/usr/bin/env python3
"""
Real-World-Benchmark: ECHTE Dokumente als Kontext (natuerliche Laenge) + echte
Query, sequentiell ueber mehrere provider@modelid (aus models.json, gitignored).

KEINE synthetischen 2k/4k-Fenster: jedes Dokument ist ein realer Kontextpunkt,
die x-Achse ist die GEMELDETE prompt_tokens (usage) — die echte Kontextlaenge.

Pro (Modell, Dokument): ein nicht-streamender Aufruf:
  tok_per_s = completion_tokens / latency_s   (effektiv, End-to-End, inkl. Prefill)
  prompt_tokens = echter Kontext (laut API)

Modelle aus testsuite/models.json (nicht eingecheckt):
  bearer: "EMPTY"/"" -> kein Auth; "$VAR"/"env:VAR" -> Env (.env wird geladen).

Beispiel:
  uv run python testsuite/bench_realworld.py --config testsuite/models.json \
      --docs ~/code/kontext.one/python-utils/packages/agentos/docs/preisblätter \
      --reasoning none --limit 7 --out runs/realworld.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from bench_openai import client  # noqa: E402

DEFAULT_QUERY = ("Du pruefst das angehaengte Vergabe-Dokument. Erstelle eine strukturierte "
                 "Zusammenfassung: Titel, Bieter/Firma, Gesamtpreis, Anzahl der Positionen "
                 "und eine kurze Einschaetzung, ob die Arithmetik stimmt. Antworte ausfuehrlich.")
_DOC_DEFAULT = os.path.join(os.path.dirname(__file__), "fixtures", "docs")
_EXCLUDE = (".venv", "node_modules", ".pytest_cache", "egg-info", "build", ".git")


def load_dotenv() -> None:
    cands = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
             os.path.abspath(".env")]
    for p in dict.fromkeys(cands):
        if os.path.isfile(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()
            return


def resolve_bearer(b: str) -> str:
    if not b:
        return ""
    b = b.strip()
    if b.upper() == "EMPTY":
        return ""
    if b.startswith("$"):
        return os.getenv(b[1:], "")
    if b.startswith("env:"):
        return os.getenv(b[4:], "")
    return b


def load_targets(args: argparse.Namespace) -> list[dict]:
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        out = []
        for m in cfg.get("models", []):
            if m.get("enabled", True) is False:
                print(f"  (überspringe deaktiviert: {m.get('name', m.get('model_id'))})")
                continue
            mid = m["model_id"]
            out.append({"name": m.get("name", mid), "model_id": mid,
                        "base_url": m["base_url"], "bearer": resolve_bearer(m.get("bearer", ""))})
        return out
    if not args.models:
        sys.exit("--config oder --models erforderlich")
    auth = resolve_bearer(args.bearer)
    return [{"name": m, "model_id": m, "base_url": args.base_url, "bearer": auth}
            for m in (x.strip() for x in args.models.split(",")) if m]


def expand_docs(spec: str, limit: int) -> list[Path]:
    if os.path.isdir(spec):
        root = Path(spec)
        files = [f for ext in ("*.md", "*.txt") for f in root.rglob(ext)
                 if not any(s in str(f) for s in _EXCLUDE)]
    else:
        files = [Path(p.strip()) for p in spec.split(",") if p.strip()]
    files = sorted((f for f in files if f.is_file()), key=lambda f: f.stat().st_size)
    if limit and limit < len(files):  # gleichmaessig ueber die Groessen streuen
        idx = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
        files = [files[i] for i in dict.fromkeys(idx)]
    return files


def _call(base: str, auth: str, model: str, reasoning: dict, doc_text: str,
          query: str, max_tokens: int) -> tuple[str, float, int | None, int | None]:
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    body = {"model": model,
            "messages": [{"role": "user", "content": f"<DOKUMENT>\n{doc_text}\n</DOKUMENT>\n\n{query}"}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(reasoning)
    with client(timeout=1800.0) as c:
        t0 = time.perf_counter()
        r = c.post(f"{base}/chat/completions", headers=H, json=body)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json()
    ch = j.get("choices", [{}])[0]
    content = ((ch.get("message") or {}).get("content", "") or "").strip()
    u = j.get("usage", {}) or {}
    return content, dt, u.get("prompt_tokens"), u.get("completion_tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-World-Benchmark: echte Docs, echte Kontextlaenge")
    ap.add_argument("--config", default=os.getenv("MODELS_CONFIG", ""))
    ap.add_argument("--base-url", default=os.getenv("BASEURL", "https://localhost:8123/v1"))
    ap.add_argument("--bearer", default=os.getenv("BEARER", ""))
    ap.add_argument("--models", default=os.getenv("MODELS", ""))
    ap.add_argument("--reasoning", default=os.getenv("REASONING", ""))
    ap.add_argument("--docs", default=os.getenv("DOCS", _DOC_DEFAULT),
                   help="Verzeichnis (glob *.md/*.txt) oder kommagetrennte Dateien")
    ap.add_argument("--query", default=os.getenv("QUERY", DEFAULT_QUERY))
    ap.add_argument("--gen-tokens", type=int, default=int(os.getenv("GEN_TOKENS", "512")))
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")), help="max Docs (gestreut nach Groesse)")
    ap.add_argument("--out", default=os.getenv("OUT", "runs/realworld.jsonl"))
    args = ap.parse_args()

    load_dotenv()
    targets = load_targets(args)
    reason = {"reasoning_effort": args.reasoning} if args.reasoning else {}
    docs = expand_docs(args.docs, args.limit)

    print(f"docs: {len(docs)}  (aus '{args.docs}')  targets={len(targets)}  "
          f"reasoning={args.reasoning or 'default'}  gen={args.gen_tokens}")
    print("  " + " | ".join(f"{f.name} ({f.stat().st_size//1024}KB)" for f in docs[:12]) +
          (" …" if len(docs) > 12 else ""))

    rows: list[tuple] = []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for t in targets:
            name, model, base, auth = t["name"], t["model_id"], t["base_url"], t["bearer"]
            print(f"\n### {name}  ({model} @ {base})")
            print(f"  {'Dokument':46} {'in_tok':>8} {'tok/s':>7} {'out':>5} {'lat_s':>6}")
            for f in docs:
                doc_text = f.read_text(encoding="utf-8", errors="ignore")
                rec = {"model_id": model, "name": name, "doc": str(f), "query": args.query}
                try:
                    ans, dt, pt, ct = _call(base, auth, model, reason, doc_text, args.query, args.gen_tokens)
                    tps = round(ct / dt, 1) if (ct and dt) else None
                    rec.update({"prompt_tokens": pt, "completion_tokens": ct, "tok_per_s": tps,
                                "latency_s": round(dt, 2), "answer": ans})
                    rows.append((name, model, f.name, pt, tps, ct, dt))
                    print(f"  {f.name:46} {(pt or 0):>8} {(tps or 0):>7.0f} {(ct or 0):>5} {dt:>6.1f}")
                except Exception as e:  # noqa: BLE001
                    rec.update({"prompt_tokens": None, "completion_tokens": None, "tok_per_s": None,
                                "latency_s": None, "answer": f"<error: {type(e).__name__}: {str(e)[:100]}>"})
                    print(f"  {f.name:46}  FEHLER: {type(e).__name__}: {str(e)[:60]}")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    print("\n=== tok/s vs Kontextlaenge (sortiert nach prompt_tokens) ===")
    print(f"  {'in_tok':>7}  {'tok/s':>6}  {'out':>4}  {'lat_s':>6}  {'modell':22}  dok")
    for name, model, fname, pt, tps, ct, dt in sorted(rows, key=lambda r: (r[3] or 0)):
        if pt is None:
            print(f"  {'-':>7}  {'-':>6}  {'-':>4}  {'-':>6}  {name[:22]:22}  {fname} (Fehler)")
            continue
        print(f"  {pt:>7}  {(tps or 0):>6.0f}  {(ct or 0):>4}  {dt:>6.1f}  {name[:22]:22}  {fname}")

    print(f"\nJSONL -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
