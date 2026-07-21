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

Modelle (zwei Wege):
  1) einzelnes Modell via CLI (wie llama-benchy):
     uv run python testsuite/bench_realworld.py \
       --base-url https://amd1.mooo.com:8123/v1 --api-key $PROXY_KEY \
       --model tu@qwen-3.6-35b \
       --docs testsuite/fixtures/docs/realworld --reasoning none
  2) mehrere Modelle via Config (Batchmodus):
     uv run python testsuite/bench_realworld.py --config testsuite/models.json \
       --docs testsuite/fixtures/docs/realworld --reasoning none
  bearer: "EMPTY"/"" -> kein Auth; "$VAR"/"env:VAR" -> Env (.env wird geladen).

Beispiel:
  uv run python testsuite/bench_realworld.py --config testsuite/models.json \
      --docs ~/code/kontext.one/python-utils/packages/agentos/docs/preisblätter \
      --reasoning none --limit 7 --out runs/realworld.jsonl

Known limitation — Tokenizer / think-Spalte:
  - in_tok/out_tok kommen aus der API (usage.prompt_tokens / completion_tokens)
    => serverseitig exakt, kein lokaler Tokenizer noetig. Auch im Stream-Modus
    seit dem proxy streaming-usage fix (0.6.17): der terminal usage-chunk wird
    durchgereicht, sodass ct echt ist (nur noch Fallback auf ÷3.5-Schaetzung,
    wenn ein Provider gar keinen usage-chunk emittet).
  - Die think-Spalte zaehlt reasoning_content (vom Server geliefert, aber nicht in
    usage aufgespalten: completion_tokens_details ist null). Da weder tiktoken
    (OpenAI-only Vokabular, falsch fuer Qwen) noch transformers (korrekt, aber
    ~500MB) hier als Abhaengigkeit vorliegen, wird geschaetzt:
        think_tok ~ len(reasoning_content) / 3.5
    (3.5 statt 4.0, weil Deutsch/dichte Tokenlaeufe). Wer exakte Werte braucht:
    HF `tokenizers` (Rust, leicht) auf das gecachte tokenizer.json anwenden —
    nicht tiktoken, nicht transformers.
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
    base = os.path.dirname(__file__)
    cands = [os.path.abspath(os.path.join(base, "..", ".env")),
             os.path.abspath(os.path.join(base, ".env")),
             os.path.abspath(".env"),
             os.path.abspath(os.path.join(base, "..", ".env.local")),
             os.path.abspath(os.path.join(base, ".env.local")),
             os.path.abspath(".env.local")]
    for p in dict.fromkeys(cands):
        if not os.path.isfile(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()


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


def resolve_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    if u.startswith("$"):
        return os.getenv(u[1:], "")
    if u.startswith("env:"):
        return os.getenv(u[4:], "")
    return u


def load_targets(args: argparse.Namespace) -> list[dict]:
    out: list[dict] = []
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        for m in cfg.get("models", []):
            if m.get("enabled", True) is False:
                print(f"  (überspringe deaktiviert: {m.get('name', m.get('model_id'))})")
                continue
            mid = m["model_id"]
            out.append({"name": m.get("name", mid), "model_id": mid,
                        "base_url": resolve_url(m["base_url"]), "bearer": resolve_bearer(m.get("bearer", ""))})
    if args.model:
        out.append({"name": args.model, "model_id": args.model,
                    "base_url": resolve_url(args.base_url), "bearer": resolve_bearer(args.bearer)})
    if not out:
        sys.exit("--model oder --config erforderlich (siehe --help)")
    return out


def expand_docs(spec: str, limit: int) -> list[Path]:
    if os.path.isdir(spec):
        root = Path(spec)
        files = [f for ext in ("*.md", "*.txt") for f in root.rglob(ext)
                 if not any(s in str(f) for s in _EXCLUDE)]
    else:
        files = [Path(p.strip()) for p in spec.split(",") if p.strip()]
    files = sorted((f for f in files if f.is_file()), key=lambda f: f.stat().st_size)
    if limit and limit < len(files):  # gleichmaessig ueber die Groessen streuen
        if limit == 1:
            files = [files[0]]
        else:
            idx = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
            files = [files[i] for i in dict.fromkeys(idx)]
    return files


def _call(base: str, auth: str, model: str, reasoning: dict, doc_text: str,
          query: str, max_tokens: int):
    """Liefert (answer, reasoning, latency_s, prompt_tokens, completion_tokens,
    finish_reason, status, error). reasoning = Inhalt von reasoning_content (oder "")."""
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    body = {"model": model,
            "messages": [{"role": "user", "content": f"<DOKUMENT>\n{doc_text}\n</DOKUMENT>\n\n{query}"}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(reasoning)
    try:
        with client(timeout=1800.0) as c:
            t0 = time.perf_counter()
            r = c.post(f"{base}/chat/completions", headers=H, json=body)
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                return None, "", round(dt, 2), None, None, None, r.status_code, r.text[:200]
            j = r.json()
    except Exception as e:  # noqa: BLE001
        return None, "", None, None, None, None, None, f"{type(e).__name__}: {str(e)[:160]}"
    ch = j.get("choices", [{}])[0]
    msg = ch.get("message") or {}
    content = (msg.get("content", "") or "").strip()
    reasoning = (msg.get("reasoning_content", "") or "").strip()
    u = j.get("usage", {}) or {}
    return content, reasoning, round(dt, 2), u.get("prompt_tokens"), u.get("completion_tokens"), ch.get("finish_reason"), 200, None


def _call_stream(base: str, auth: str, model: str, reasoning: dict, doc_text: str,
                 query: str, max_tokens: int):
    """Streaming-Variante von _call. Liefert dieselben Felder plus ttft (Time-to-first-token).
    (answer, reasoning, latency_s, prompt_tokens, completion_tokens, finish_reason, status, error, ttft)."""
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    body = {"model": model,
            "messages": [{"role": "user", "content": f"<DOKUMENT>\n{doc_text}\n</DOKUMENT>\n\n{query}"}],
            "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(reasoning)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    first = None
    pt = ct = finish = status = None
    err = None
    try:
        with client(timeout=1800.0) as c:
            t0 = time.perf_counter()
            with c.stream("POST", f"{base}/chat/completions", headers=H, json=body) as s:
                if s.status_code != 200:
                    dt = time.perf_counter() - t0
                    return None, "", round(dt, 2), None, None, None, s.status_code, s.read().decode(errors="ignore")[:200], None
                for line in s.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    try:
                        d = json.loads(line[6:])
                    except Exception:  # noqa: BLE001
                        continue
                    delta = (d.get("choices") or [{}])[0].get("delta", {}) or {}
                    if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                        first = time.perf_counter()
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    ch = (d.get("choices") or [{}])[0]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    u = d.get("usage")
                    if u:
                        pt = u.get("prompt_tokens")
                        ct = u.get("completion_tokens")
            dt = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return None, "", None, None, None, None, None, f"{type(e).__name__}: {str(e)[:160]}", None
    ttft = round(first - t0, 2) if first else None
    content = "".join(content_parts).strip()
    reasoning = "".join(reasoning_parts).strip()
    # ct/prompt_tokens kommen jetzt i.d.R. echt aus dem terminalen usage-chunk
    # (proxy fix 0.6.17 gibt stream_options.include_usage durch). Die ÷3.5-Heuristik
    # ist nur noch Fallback fuer Provider, die keinen usage-chunk emitten.
    if ct is None and (content or reasoning):
        ct = round((len(content) + len(reasoning)) / 3.5)
    return content, reasoning, round(dt, 2), pt, ct, finish, 200, None, ttft


def check_answer(answer: str | None, expected: dict | None) -> str:
    """Prueft answer gegen expected.answer_contains (set/exact) -> '✓'/'✗'/'-'.
    '-' = kein expected definiert (kein Case-Treffer)."""
    if not expected:
        return "-"
    if not answer:
        return "✗"
    needles = expected.get("answer_contains", [])
    if not needles:
        return "-"
    ok = all(n in answer for n in needles)
    return "✓" if ok else "✗"


def load_cases_for_docs(cases_path: str, doc_names: list[str]) -> dict[str, dict]:
    """Mapt doc-basename -> case (mit query+expected). Liefert {} falls kein cases-File."""
    if not cases_path or not os.path.isfile(cases_path):
        return {}
    out: dict[str, dict] = {}
    for line in open(cases_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        for d in c.get("docs", []):
            out[os.path.basename(d)] = c
    return {dn: out[dn] for dn in doc_names if dn in out}


def md_table(headers: list[str], rows: list[list[str]],
               align: list[str] | None = None) -> str:
    """Markdown-Tabelle mit spaltenbuendig ausgerichteten Zellen (Padding)."""
    n = len(headers)
    align = align or ["left"] * n
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r[:n]):
            widths[i] = max(widths[i], len(cell))
    def pad(cell: str, i: int) -> str:
        a = align[i]
        if a == "right":
            return cell.rjust(widths[i])
        if a == "center":
            return cell.center(widths[i])
        return cell.ljust(widths[i])
    def sep(i: int) -> str:
        a = align[i]
        dashes = "-" * widths[i]
        if a == "right":
            return f"{dashes}:"
        if a == "center":
            return f":{dashes}:"
        return f":{dashes}"
    lines = ["| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |",
             "| " + " | ".join(sep(i) for i in range(n)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(pad(c, i) for i, c in enumerate(r[:n])) + " |")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    _default_base = (os.getenv("BASEURL") or os.getenv("AMD_BASEURL")
                     or f"{os.getenv('PROXY_SCHEME', 'https')}://{os.getenv('PROXYHOST', 'localhost')}:{os.getenv('PROXY_PORT', '8123')}/v1")
    ap = argparse.ArgumentParser(description="Real-World-Benchmark: echte Docs, echte Kontextlaenge")
    ap.add_argument("--config", default=os.getenv("MODELS_CONFIG", ""))
    ap.add_argument("--base-url", default=_default_base)
    ap.add_argument("--model", default=os.getenv("MODEL", ""),
                   help="einzelnes Modell (provider@model_id). Fuer mehrere: --config")
    ap.add_argument("--bearer", "--api-key", dest="bearer",
                   default=os.getenv("BEARER") or os.getenv("PROXY_KEY") or os.getenv("AMD_TOKEN", ""))
    ap.add_argument("--reasonings", default=os.getenv("REASONINGS", "nothink,think"),
                   help="Komma-Liste Denk-Modi: nothink=ohne, think=mit (alias: none/high)")
    ap.add_argument("--docs", default=os.getenv("DOCS", _DOC_DEFAULT),
                   help="Verzeichnis (glob *.md/*.txt) oder kommagetrennte Dateien")
    ap.add_argument("--cases", default=os.getenv("CASES", os.path.join(os.path.dirname(_DOC_DEFAULT), "cases", "realworld_lv.jsonl")),
                   help="JSONL mit doc->query+expected (fuer check-Spalte). Leer = kein check.")
    ap.add_argument("--query", default=os.getenv("QUERY", DEFAULT_QUERY))
    ap.add_argument("--gen-tokens", type=int, default=int(os.getenv("GEN_TOKENS", "2048")))
    ap.add_argument("--streams", default=os.getenv("STREAMS", "nostrm"),
                   help="Komma-Liste Stream-Modi: nostrm=ohne, strm=mit (alias: none/stream)")
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")), help="max Docs (gestreut nach Groesse)")
    ap.add_argument("--out", default=os.getenv("OUT", "runs/realworld.jsonl"))
    args = ap.parse_args()

    targets = load_targets(args)
    _MODE_ALIASES = {"none": "nothink", "off": "nothink", "nothink": "nothink",
                     "high": "think", "on": "think", "think": "think",
                     "low": "low", "default": "default"}
    _STREAM_ALIASES = {"none": "nostrm", "nostrm": "nostrm",
                      "stream": "strm", "strm": "strm"}
    modes = [_MODE_ALIASES.get(m.strip(), m.strip()) for m in args.reasonings.split(",") if m.strip()]
    streams = [_STREAM_ALIASES.get(s.strip(), s.strip()) for s in args.streams.split(",") if s.strip()]
    docs = expand_docs(args.docs, args.limit)
    case_map = load_cases_for_docs(args.cases, [f.name for f in docs])

    print(f"docs: {len(docs)}  (aus '{args.docs}')  targets={len(targets)}  "
          f"reasonings={modes}  streams={streams}  gen={args.gen_tokens}  cases: {len(case_map)}/{len(docs)}")
    print("  " + " | ".join(f"{f.name} ({f.stat().st_size//1024}KB)" for f in docs[:12]) +
          (" …" if len(docs) > 12 else ""))

    rows: list[tuple] = []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for t in targets:
            name, model, base, auth = t["name"], t["model_id"], t["base_url"], t["bearer"]
            print(f"\n### {name}  ({model} @ {base})")
            print(f"  {'Dokument':34} {'mode':5} {'strm':5} {'in_tok':>7} {'tok/s':>6} {'out':>5} {'think':>5} {'ttft':>5} {'lat':>5} {'fin':>5} {'chk':>3} {'x':>2}")
            for f in docs:
                doc_text = f.read_text(encoding="utf-8", errors="ignore")
                case = case_map.get(f.name)
                query = case["query"] if case else args.query
                for mode in modes:
                    if mode == "think":
                        reason = {"reasoning_effort": "high"}
                        mt = max(args.gen_tokens, 8192)
                    elif mode == "nothink":
                        reason = {"reasoning_effort": "none"}
                        mt = args.gen_tokens
                    elif mode in ("low",):
                        reason = {"reasoning_effort": mode}
                        mt = max(args.gen_tokens, 8192)
                    else:
                        reason = {}
                        mt = args.gen_tokens
                    for strm in streams:
                        do_stream = strm == "strm"
                        if do_stream:
                            ans, rsn, dt, pt, ct, finish, status, err, ttft = _call_stream(base, auth, model, reason, doc_text, query, mt)
                        else:
                            ans, rsn, dt, pt, ct, finish, status, err = _call(base, auth, model, reason, doc_text, query, mt)
                            ttft = None
                        tps = round(ct / dt, 1) if (ct and dt) else None
                        think_tok = round(len(rsn) / 3.5) if rsn else 0
                        chk = check_answer(ans, case.get("expected") if case else None)
                        x = "x" if (finish == "length" or (status and status != 200)) else ""
                        rec = {"model_id": model, "name": name, "doc": str(f), "query": query,
                               "reasoning": mode, "stream": strm, "prompt_tokens": pt, "completion_tokens": ct,
                               "tok_per_s": tps, "latency_s": dt, "ttft": ttft, "finish_reason": finish,
                               "status_code": status, "error": err, "answer": ans,
                               "reasoning_content": rsn, "think_tokens_est": think_tok,
                               "check": chk, "expected": case.get("expected") if case else None}
                        rows.append((name, model, f.name, mode, strm, pt, tps, ct, think_tok, ttft, dt, finish, status, chk, x))
                        if status and status != 200:
                            print(f"  {f.name:34} {mode:5} {strm:5}  HTTP {status}")
                        else:
                            ttft_s = f"{ttft:.1f}" if ttft else "-"
                            print(f"  {f.name:34} {mode:5} {strm:5} {(pt or 0):>7} {(tps or 0):>6.0f} "
                                  f"{(ct or 0):>5} {think_tok:>5} {ttft_s:>5} {dt or 0:>5.1f} {(finish or '-')[:5]:>5} {chk:>3} {x:>2}")
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()

    # --- markdown result table (llama-benchy-style, spaltenbuendig) ---
    print("\n" + "=" * 72)
    print("### Results\n")
    results_rows = []
    for name, model, fname, mode, strm, pt, tps, ct, think_tok, ttft, dt, finish, status, chk, x in sorted(rows, key=lambda r: (r[0], r[3], r[4], r[5] or 0)):
        failed = status is not None and status != 200
        if failed:
            results_rows.append([name, fname, mode, strm, "-", "-", "-", "-", "-", f"HTTP {status}", "✗", "x"])
        else:
            ttft_s = f"{ttft:.1f}" if ttft else "-"
            results_rows.append([name, fname, mode, strm, str(pt or "~"), str(tps or "-"), str(ct or "-"), str(think_tok), ttft_s, (finish or "-")[:6], chk, x])
    print(md_table(["model", "document", "mode", "strm", "in_tok", "tok/s", "out_tok", "think", "ttft", "finish", "check", "x"],
                   results_rows, align=["left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "center", "center"]))

    # --- per-model/mode/stream summary ---
    by_key: dict[tuple, list[tuple]] = {}
    for name, model, fname, mode, strm, pt, tps, ct, think_tok, ttft, dt, finish, status, chk, x in rows:
        if status is not None and status != 200:
            continue
        by_key.setdefault((name, mode, strm), []).append((pt or 0, tps or 0, dt, finish, chk, ttft))
    if by_key:
        print("\n### Summary (per model × mode × stream)\n")
        sum_rows = []
        for (name, mode, strm), recs in by_key.items():
            tps_vals = [r[1] for r in recs]
            avg_tps = sum(tps_vals) / len(tps_vals)
            recs_sorted = sorted(recs, key=lambda r: r[0])
            truncated = sum(1 for r in recs if r[3] == "length")
            passed = sum(1 for r in recs if r[4] == "✓")
            ttfts = [r[5] for r in recs if r[5]]
            avg_ttft = f"{sum(ttfts)/len(ttfts):.1f}" if ttfts else "-"
            flag = f" ({truncated}×length)" if truncated else ""
            sum_rows.append([f"{name} [{mode}/{strm}]", str(len(recs)), f"{avg_tps:.0f}",
                            f"{recs_sorted[0][1]:.0f}", f"{recs_sorted[-1][1]:.0f}",
                            f"{sum(r[2] for r in recs)/len(recs):.1f}" + flag, avg_ttft,
                            f"{passed}/{len(recs)}"])
        print(md_table(["model [mode/strm]", "docs", "avg tok/s", "tok/s @ min ctx", "tok/s @ max ctx", "avg lat_s", "avg ttft", "check"],
                       sum_rows, align=["left", "right", "right", "right", "right", "left", "right", "center"]))

    print(f"\nbench_realworld date: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
          f"targets: {len(targets)} | docs: {len(docs)} | reasonings: {','.join(modes)} | streams: {','.join(streams)}")
    print(f"JSONL -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
