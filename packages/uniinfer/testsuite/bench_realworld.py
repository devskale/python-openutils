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
    durchgereicht.
  - reasoning_tokens wird EXAKT aus completion_tokens_details.reasoning_tokens
    gelesen, wenn der Server ihn liefert (vLLM stream w/ include_usage, openrouter
    beide Modi). Nur wenn er fehlt (vLLM non-stream, groq) wird aus der Laenge des
    reasoning_content geschaetzt (÷3.5) und reasoning_tokens_estimated=True gesetzt.
    ('~' in der think-Spalte = geschaetzt.) vLLM non-stream emitiert details gar
    nicht (TU-Realitaet, kein proxy-Bug).
  - content_tokens = completion_tokens − reasoning_tokens (abgeleitet).
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

# Heuristic divisor for estimating token counts from text length (German /
# dense runs). Used only when the server provides no exact field.
_REASONING_TOKEN_GUESS_DIVISOR = 3.5


def extract_usage(usage: dict | None, reasoning_text: str = "") -> dict:
    """Extract token counts from a server usage object (the bench's usage seam).

    reasoning_tokens precedence:
      1. exact ``completion_tokens_details.reasoning_tokens`` when the server
         provides it (vLLM stream w/ include_usage, openrouter both modes);
      2. estimated from reasoning_content text length (÷3.5) when text is
         present but no details (vLLM non-stream, groq — details never emitted);
      3. None otherwise.

    content_tokens = completion − reasoning when both known, else None.
    """
    u = usage or {}
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    det = u.get("completion_tokens_details") or {}
    rt = det.get("reasoning_tokens")
    estimated = False
    if rt is None and reasoning_text:
        rt = round(len(reasoning_text) / _REASONING_TOKEN_GUESS_DIVISOR)
        estimated = True
    # reasoning is a subset of completion — cap an over-estimate at ct so
    # content_tokens never goes negative.
    if rt is not None and ct is not None and rt > ct:
        rt = ct
    content = (ct - rt) if (ct is not None and rt is not None) else None
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "reasoning_tokens_estimated": estimated,
        "content_tokens": content,
    }


_EMPTY_USAGE = {"prompt_tokens": None, "completion_tokens": None,
                "reasoning_tokens": None, "reasoning_tokens_estimated": False,
                "content_tokens": None}


def _result(content, reasoning, latency, usage, finish, status, error, ttft=None) -> dict:
    """Build a uniform result dict from a call path."""
    u = usage if usage else dict(_EMPTY_USAGE)
    return {"content": content, "reasoning": reasoning, "latency": latency,
            "finish_reason": finish, "status": status, "error": error, "ttft": ttft, **u}


# Minimum decode-phase duration for a trustworthy decode tok/s. Below this the
# measurement is noise (lat ≈ ttft for short outputs → divide by ~0).
_DECODE_MIN_PHASE_S = 1.0

# Minimum max_tokens for reasoning modes. Thinking models spend the budget on
# reasoning before the visible answer; too low a cap truncates the answer
# (finish=length) and the measured duration is time-to-cap, not time-to-task.
# Matches the proxy chat-path default (32768) so reasoning tasks complete.
_THINK_MIN_TOKENS = 32768


def compute_throughput(completion_tokens, latency, ttft) -> dict:
    """Throughput metrics: effective (incl prefill) and decode (generation only).

    effective = completion / latency        (end-to-end, incl prefill)
    decode    = completion / (latency - ttft) (generation phase only)

    decode is None when it can't be measured reliably: non-stream (no ttft),
    lat ≤ ttft, or the decode phase is shorter than _DECODE_MIN_PHASE_S (short
    outputs where lat ≈ ttft make the rate divide by rounding noise).
    """
    eff = round(completion_tokens / latency, 1) if (completion_tokens and latency) else None
    dec = None
    if completion_tokens and latency and ttft and latency > ttft:
        phase = latency - ttft
        if phase >= _DECODE_MIN_PHASE_S:
            dec = round(completion_tokens / phase, 1)
    return {"effective": eff, "decode": dec}


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


def _collect_nonstream(c, url, H, body):
    """Single-read collector. Returns (content, reasoning, dt, usage_obj,
    finish, status, error, ttft)."""
    t0 = time.perf_counter()
    r = c.post(url, headers=H, json=body)
    dt = round(time.perf_counter() - t0, 2)
    if r.status_code != 200:
        return None, "", dt, None, None, r.status_code, r.text[:200], None
    j = r.json()
    ch = j.get("choices", [{}])[0]
    msg = ch.get("message") or {}
    content = (msg.get("content", "") or "").strip()
    rsn = (msg.get("reasoning_content", "") or "").strip()
    return content, rsn, dt, j.get("usage"), ch.get("finish_reason"), 200, None, None


def _collect_stream(c, url, H, body):
    """SSE collector. Same return shape as _collect_nonstream, plus ttft.
    Keeps the full usage object (not just pt/ct) so extract_usage can read
    completion_tokens_details.reasoning_tokens when vLLM emits it."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    first = usage_obj = finish = None
    t0 = time.perf_counter()
    with c.stream("POST", url, headers=H, json=body) as s:
        if s.status_code != 200:
            dt = round(time.perf_counter() - t0, 2)
            return None, "", dt, None, None, s.status_code, \
                s.read().decode(errors="ignore")[:200], None
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
                usage_obj = u  # terminal chunk carries full usage incl details
        dt = round(time.perf_counter() - t0, 2)
    ttft = round(first - t0, 2) if first else None
    content = "".join(content_parts).strip()
    rsn = "".join(reasoning_parts).strip()
    # Fallback: Server emitiert gar keinen usage-chunk -> ct aus Text schaetzen.
    if usage_obj is None and (content or rsn):
        usage_obj = {"completion_tokens": round((len(content) + len(rsn)) / _REASONING_TOKEN_GUESS_DIVISOR)}
    return content, rsn, dt, usage_obj, finish, 200, None, ttft


def _call(base: str, auth: str, model: str, reasoning: dict, doc_text: str,
          query: str, max_tokens: int, stream: bool = False, timeout: float = 1800.0):
    """One call path; branches on ``stream``. The shared request (headers,
    body, the _result + extract_usage wrapping) lives here; the two collectors
    own only the HTTP-shape difference (single read vs SSE iteration)."""
    H = {"Content-Type": "application/json"}
    if auth:
        H["Authorization"] = f"Bearer {auth}"
    body = {"model": model,
            "messages": [{"role": "user", "content": f"<DOKUMENT>\n{doc_text}\n</DOKUMENT>\n\n{query}"}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(reasoning)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    url = f"{base}/chat/completions"
    try:
        with client(timeout=timeout) as c:
            collector = _collect_stream if stream else _collect_nonstream
            content, rsn, dt, usage, finish, status, err, ttft = collector(c, url, H, body)
    except Exception as e:  # noqa: BLE001
        return _result(None, "", None, None, None, None, f"{type(e).__name__}: {str(e)[:160]}")
    return _result(content, rsn, dt, extract_usage(usage, rsn), finish, status, err, ttft)


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


def load_cases_for_docs(cases_path: str, doc_names: list[str]) -> dict[str, list[dict]]:
    """Mapt doc-basename -> Liste der cases (jeder = ein ö-Vergaberecht-Test mit
    query+expected). Mehrere Tests pro Doc, damit ein Lauf den ganzen Raum deckt.
    Liefert {} falls kein cases-File."""
    if not cases_path or not os.path.isfile(cases_path):
        return {}
    out: dict[str, list[dict]] = {}
    for line in open(cases_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        for d in c.get("docs", []):
            out.setdefault(os.path.basename(d), []).append(c)
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
    ap.add_argument("--max-cases", type=int, default=int(os.getenv("MAX_CASES", "0")),
                    help="max cases (oesterreich-Vergaberecht-Tests) pro Doc (0 = alle). "
                         "Klein setzen fuer schnelle Laufe, z.B. --max-cases 1.")
    ap.add_argument("--timeout", type=float, default=float(os.getenv("TIMEOUT", "1800")),
                    help="httpx read timeout per call in seconds (default 1800 = 30 min). "
                         "Raise for slow local models, e.g. --timeout 7200. Use --streams strm "
                         "so the timeout applies per-chunk, not to the whole response.")
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
            print(f"  {'Dokument':34} {'mode':7} {'tok/s':>6} {'ttft':>5} {'total':>6}")
            for f in docs:
                doc_text = f.read_text(encoding="utf-8", errors="ignore")
                cases_for_doc = case_map.get(f.name) or [None]  # None = default query
                if args.max_cases:
                    cases_for_doc = cases_for_doc[:args.max_cases]
                for case in cases_for_doc:
                    query = case["query"] if case else args.query
                    for mode in modes:
                        if mode == "think":
                            reason = {"reasoning_effort": "high"}
                            mt = max(args.gen_tokens, _THINK_MIN_TOKENS)
                        elif mode == "nothink":
                            reason = {"reasoning_effort": "none"}
                            mt = args.gen_tokens
                        elif mode in ("low",):
                            reason = {"reasoning_effort": mode}
                            mt = max(args.gen_tokens, _THINK_MIN_TOKENS)
                        else:
                            reason = {}
                            mt = args.gen_tokens
                        for strm in streams:
                            do_stream = strm == "strm"
                            res = _call(base, auth, model, reason, doc_text, query, mt, stream=do_stream, timeout=args.timeout)
                            ans = res["content"]; rsn = res["reasoning"]; dt = res["latency"]
                            pt = res["prompt_tokens"]; ct = res["completion_tokens"]
                            rt = res["reasoning_tokens"]; rt_est = res["reasoning_tokens_estimated"]
                            content_tok = res["content_tokens"]
                            finish = res["finish_reason"]; status = res["status"]; err = res["error"]; ttft = res["ttft"]
                            tps = round(ct / dt, 1) if (ct and dt) else None
                            dec_tps = compute_throughput(ct, dt, ttft)["decode"]  # logged only
                            think_display = (f"{rt}{'~' if rt_est else ''}" if rt is not None else "-")
                            chk = check_answer(ans, case.get("expected") if case else None)
                            rec = {"model_id": model, "name": name, "doc": str(f), "query": query,
                                   "reasoning": mode, "stream": strm, "prompt_tokens": pt, "completion_tokens": ct,
                                   "reasoning_tokens": rt, "reasoning_tokens_estimated": rt_est,
                                   "content_tokens": content_tok,
                                   "tok_per_s": tps, "tok_per_s_decode": dec_tps,
                                   "latency_s": dt, "ttft": ttft, "finish_reason": finish,
                                   "status_code": status, "error": err, "answer": ans,
                                   "reasoning_content": rsn,
                                   "check": chk, "expected": case.get("expected") if case else None}
                            rows.append((name, mode, strm, f.name, tps, ttft, dt, status))
                            if status and status != 200:
                                print(f"  {f.name:34} {mode:7}  HTTP {status}")
                            else:
                                ttft_s = f"{ttft:.1f}" if ttft else "-"
                                print(f"  {f.name:34} {mode:7} {(tps or 0):>6.0f} {ttft_s:>5} {dt or 0:>6.1f}")
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()

    # --- markdown result table (llama-benchy-style, spaltenbuendig) ---
    print("\n" + "=" * 72)
    print("### Results\n")
    results_rows = []
    for name, mode, strm, fname, tps, ttft, dt, status in sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3])):
        if status is not None and status != 200:
            results_rows.append([name, fname, mode, "-", "-", f"HTTP {status}"])
        else:
            ttft_s = f"{ttft:.1f}" if ttft else "-"
            time_s = f"{dt:.1f}" if dt else "-"
            results_rows.append([name, fname, mode, str(tps or "-"), ttft_s, time_s])
    print(md_table(["model", "document", "mode", "tok/s", "ttft", "total"],
                   results_rows, align=["left", "left", "left", "right", "right", "right"]))

    # --- per-model/mode/stream summary ---
    by_key: dict[tuple, list[tuple]] = {}
    for name, mode, strm, fname, tps, ttft, dt, status in rows:
        if status is not None and status != 200:
            continue
        by_key.setdefault((name, mode, strm), []).append((tps or 0, ttft, dt))
    if by_key:
        print("\n### Summary (per model × mode)\n")
        sum_rows = []
        for (name, mode, strm), recs in by_key.items():
            tps_vals = [r[0] for r in recs if r[0]]
            avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
            ttfts = [r[1] for r in recs if r[1]]
            avg_ttft = f"{sum(ttfts)/len(ttfts):.1f}" if ttfts else "-"
            avg_lat = sum(r[2] for r in recs) / len(recs) if recs else 0
            sum_rows.append([f"{name} [{mode}]", str(len(recs)), f"{avg_tps:.0f}", avg_ttft, f"{avg_lat:.1f}"])
        print(md_table(["model [mode]", "runs", "avg tok/s", "avg ttft", "avg total"],
                       sum_rows, align=["left", "right", "right", "right", "right"]))

    print(f"\nbench_realworld date: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
          f"targets: {len(targets)} | docs: {len(docs)} | reasonings: {','.join(modes)} | streams: {','.join(streams)}")
    print(f"JSONL -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
