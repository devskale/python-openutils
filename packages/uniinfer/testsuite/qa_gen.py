#!/usr/bin/env python3
"""
Case-getriebener Benchmark + Antwortgenerator (OFS/pdf2md/strukt2meta/agentos).

Entdeckt ALLE Testfaelle (doc + query + expected) unter fixtures/cases/*.jsonl und
laeuft sequentiell ueber alle provider@modelid (models.json) UND mehrere
Reasoning-Modi (z.B. none=no-think, high=think) — damit think/no-think direkt
vergleichbar sind.

Pro (Modell, Case, Modus): tok_per_s (Output/Latenz), output_tokens, prompt_tokens
(echter Kontext), answer. HTTP-Fehler (402/429/…) werden erkannt und als
status_code+error ins JSONL geschrieben (kein Absturz). KEINE Bewertung hier
(expected wird mitgeschrieben; Mensch/LLM-as-Judge extern).

Beispiel:
  uv run python testsuite/qa_gen.py --config testsuite/models.json --reasonings none,high
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from bench_realworld import load_dotenv, resolve_bearer, load_targets  # noqa: E402

_DIR = os.path.dirname(__file__)
DEFAULT_CASES = os.path.join(_DIR, "fixtures", "cases")
DEFAULT_DOCS_ROOT = os.path.join(_DIR, "fixtures", "docs")


def expand_cases(spec: str) -> list[tuple[str, dict]]:
    if os.path.isdir(spec):
        files = sorted(Path(spec).rglob("*.jsonl"))
    else:
        files = [Path(spec)]
    out: list[tuple[str, dict]] = []
    for cf in files:
        if not cf.is_file():
            continue
        for line in cf.open(encoding="utf-8"):
            line = line.strip()
            if line:
                out.append((str(cf), json.loads(line)))
    return out


def _call(base: str, auth: str, model: str, reasoning: dict, doc_text: str,
          query: str, max_tokens: int):
    """Liefert (answer, latency_s, prompt_tokens, completion_tokens, status_code, error)."""
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    body = {"model": model,
            "messages": [{"role": "user", "content": f"<DOKUMENT>\n{doc_text}\n</DOKUMENT>\n\n{query}"}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(reasoning)
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=15, read=1800, write=60, pool=60), verify=False) as c:
            t0 = __import__("time").perf_counter()
            r = c.post(f"{base}/chat/completions", headers=H, json=body)
            dt = __import__("time").perf_counter() - t0
            if r.status_code != 200:
                return None, round(dt, 2), None, None, r.status_code, r.text[:200]
            j = r.json()
    except httpx.HTTPStatusError as e:
        return None, None, None, None, e.response.status_code, str(e)[:200]
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, None, f"{type(e).__name__}: {str(e)[:160]}"
    ch = j.get("choices", [{}])[0]
    content = ((ch.get("message") or {}).get("content", "") or "").strip()
    u = j.get("usage", {}) or {}
    return content, round(dt, 2), u.get("prompt_tokens"), u.get("completion_tokens"), 200, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Case-Benchmark: Modelle x Cases x (think/no-think)")
    ap.add_argument("--config", default=os.getenv("MODELS_CONFIG", ""))
    ap.add_argument("--base-url", default=os.getenv("BASEURL", "https://localhost:8123/v1"))
    ap.add_argument("--bearer", default=os.getenv("BEARER", ""))
    ap.add_argument("--models", default=os.getenv("MODELS", ""))
    ap.add_argument("--reasonings", default=os.getenv("REASONINGS", "none,high"),
                   help="Komma-Liste Reasoning-Modi: none=no-think, high=think, low, off")
    ap.add_argument("--cases", default=os.getenv("CASES", DEFAULT_CASES))
    ap.add_argument("--docs-root", default=os.getenv("DOCS_ROOT", DEFAULT_DOCS_ROOT))
    ap.add_argument("--gen-tokens", type=int, default=int(os.getenv("GEN_TOKENS", "512")))
    ap.add_argument("--app", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.getenv("OUT", "runs/qa_gen.jsonl"))
    args = ap.parse_args()

    load_dotenv()
    targets = load_targets(args)
    cases = expand_cases(args.cases)
    if args.app:
        cases = [(f, c) for f, c in cases if c.get("app") == args.app]
    if args.limit:
        cases = cases[: args.limit]
    modes = [m.strip() for m in args.reasonings.split(",") if m.strip()]
    docs_root = Path(args.docs_root)

    print(f"cases={len(cases)}  targets={len(targets)}  reasonings={modes}  gen={args.gen_tokens}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # rows[(model,case_id)] = {mode: (pt, tps, ct, dt, status)}
    grouped: dict[tuple, dict] = {}
    with open(args.out, "w", encoding="utf-8") as fout:
        for t in targets:
            name, model, base, auth = t["name"], t["model_id"], t["base_url"], t["bearer"]
            print(f"\n### {name}  ({model} @ {base})")
            for cf, c in cases:
                docs_text = "\n\n".join((docs_root / d).read_text(encoding="utf-8", errors="ignore")
                                         for d in c.get("docs", []))
                for mode in modes:
                    reason = {"reasoning_effort": mode} if mode and mode != "default" else {}
                    mt = max(args.gen_tokens, 2048) if mode in ("high", "on") else args.gen_tokens
                    ans, dt, pt, ct, status, err = _call(base, auth, model, reason, docs_text, c["query"], mt)
                    rec = {"model_id": model, "name": name, "case_id": c.get("id"), "app": c.get("app"),
                           "task": c.get("task"), "reasoning": mode, "query": c.get("query"),
                           "docs": c.get("docs", []), "expected": c.get("expected", {}),
                           "answer": ans, "tok_per_s": (round(ct / dt, 1) if (ct and dt) else None),
                           "output_tokens": ct, "prompt_tokens": pt, "latency_s": dt,
                           "status_code": status, "error": err}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    grouped.setdefault((name, c.get("id")), {})[mode] = (pt, (round(ct / dt, 1) if (ct and dt) else None), ct, dt, status)
                    cell = f"{round(ct/dt,1)} tok/s" if (ct and dt) else f"HTTP {status}"
                    print(f"  [{str(c.get('app')):9}] {str(c.get('id')):22} {mode:5} in={pt} {cell}")
    # think vs no-think Vergleich
    print("\n=== think vs no-think  (tok/s je Modus; HTTP-Code bei Fehler) ===")
    print(f"  {'in_tok':>7}  {'modell':22}  {'case':22}  " +
          "  ".join(f"{m:>8}" for m in modes))
    def _pt_key(v):
        return (v.get("none") or v.get(list(v)[0]) or (None,))[0] or 0
    for (name, cid), v in sorted(grouped.items(), key=lambda kv: _pt_key(kv[1])):
        pt = next((x[0] for x in v.values() if x[0]), None)
        cells = "  ".join((f"{v[m][1]:>8.0f}" if (m in v and v[m][1]) else
                           (f"{'HTTP'+str(v[m][4]):>8}" if m in v and v[m][4] else f"{'-':>8}")) for m in modes)
        print(f"  {(pt or 0):>7}  {name[:22]:22}  {str(cid)[:22]:22}  {cells}")
    print(f"\nJSONL -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
