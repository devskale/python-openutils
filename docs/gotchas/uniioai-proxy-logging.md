# Gotchas — uniioai-proxy: Logging-Kette & Fehlerlokalisierung

> Stand 2026-10-05, aus dem Incident „opencode@space-bunny-free → 400 (invisible)".
> Kette: pi (mac) → nginx (amd:443, vhost `uniinfer.skale.dev`) → uniioai-proxy
> (amd, **System-Unit**, :8124) → Provider (tu / opencode zen / kilo / …).

## Untersuchungsschema (Fehler-Lokalisierung über 4 Ebenen)

```
Ebene 0: pi-Client          ~/.pi/agent/sessions/<proj>/<ts>_<id>.jsonl  ← Ground Truth
Ebene 1: nginx (amd:443)    sudo grep /var/log/nginx/access.log (+ .1 bei Rotation)
Ebene 2: uniioai-proxy      sudo journalctl -u uniioai-proxy   (SYSTEM-Unit!)
Ebene 3: Provider-Raw-Logs  amd:packages/uniinfer/logs/{tu,opencode}_raw_chat.log
```

Beweisregel: Fehler sitzt auf Ebene N, wenn N+1 den Request nie gesehen hat
(Log-Silence) und eine Direkt-Probe (curl am Proxy vorbei / gegen upstream)
den Fehler dort reproduziert.

## Gotchas

1. **`uniioai-proxy` ist eine systemd-SYSTEM-Unit, keine --user-Unit.**
   `ssh amd` läuft als `ubuntu`, aber `systemctl --user status uniioai-proxy`
   → „could not be found", `journalctl --user-unit=uniioai-proxy` → leer.
   Immer: `sudo journalctl -u uniioai-proxy` / `systemctl show uniioai-proxy`.

2. **Streaming-Fehler waren lange unsichtbar (HTTP 200 + 619b).** pi streamt
   immer (`stream: true`); Provider-Fehler werden als SSE-Error-Event IN einem
   200-Response relaid. nginx zeigt nur `POST … 200 619` — kein 4xx.
   - Seit 2026-10-05 (`d5cabc1` + Nachfolger): END-Line auch für SSE
     (`… - sse`-Suffix, vorher: `if "text/event-stream" not in ct` → nie geloggt)
     und STREAM-OPEN-Audit + Upstream-non-200-Capture als **Parent-Seam in
     `OpenAICompatibleChatProvider.astream_complete`** (logging_utils:
     `audit_stream_open`/`capture_upstream_error`) — jedes Thin-Provider-Modul
     (kilo, mistral, …) erbt sie; tu behält seine eigene Instrumentierung;
     opencode loggt zusätzlich seine Spezialpfade (`_chat_acomplete`,
     responses-Dialekt). Regression-Test:
     `uniinfer/tests/test_audit_logging.py`.

3. **pi-Session-JSONL ist die Ground Truth für „welches Modell/Provider
   failte".** Grep: `"stopReason":"error"` → `errorMessage` enthält den
   Provider-Fehler inkl. Format (`OpenCode error: OpenCode API error: 403 …`
   = uniinfer-Format → Request WAR am Proxy). `"provider":"unii","model":"…"`
   steht an derselben Message.

4. **`finish_reason: error` ist ein uniinfer-synthetisierter Marker, kein
   pi-Bug.** `tu.py` yieldt bei Empty-Stream / Preemption / silent rate-limit
   shadow absichtlich `finish_reason="error"` statt zu raisen → pi zeigt
   „Provider finish_reason: error". Details: `logs/tu_raw_chat.log`.

5. **Opencode-Free-Tier: 403 FreeTierError ≠ Gate-Fail.** Das Gate (UA/IDs/
   12-Tool-Array/stream) wird passiert; der 403 kommt von der Modell-
   Verfügbarkeit („not available in your country …"). 400 `invalid_request:
   Upstream request failed` = zens Upstream selbst, payload-abhängig
   (Mini-Payloads laufen; pi-Formate teils nicht) — seit dem Raw-Log
   reproduzierbar beobachtbar.

6. **Deploy des Proxies NUR via `./deploy.sh --all-extras`** auf amd
   (`ssh amd 'cd code/python-openutils/packages/uniinfer && ./deploy.sh
   --all-extras'`) — bare `deploy.sh` strippt die Provider-SDKs. Der Proxy
   läuft from-source aus dem Checkout; Code-Änderung → commit+push → deploy.

7. **DNS/vhost:** `uniinfer.skale.dev` → 138.2.179.13 (amd public) → nginx
   vhost → `localhost:8124`. Scanner-Müll im access.log (leere Requests,
   130.61.x.x) nicht mit echten Fails verwechseln.

8. **Prod-`.env` vergiftet die Test-Suite auf Prod-Boxen** (gefunden
   2026-10-05 vom on-machine deploy-gate, erster Lauf): `provider_access.py`
   lud beim IMPORT `load_dotenv(CWD/.env, override=True)` — auf amd liegt die
   echte `.env` (Token-Allowlist, Mem-Guard) und färbte auf die Suite ab
   (13 rot auf amd, grün auf Dev-Boxen). Fix: dotenv-Laden nur außerhalb
   pytest. Merksatz: import-time-Umgebungsmutation = Suite niemals hermetisch.

9. **deploy.sh pulled seine eigene Erneuerung mid-run:** der Lauf, der die
   neue deploy.sh pullt, exekutiert noch die ALTE (bash lielt das geöffnete
   Script). Erst der nächste Lauf hat den neuen Gate — kein Bug, nur
   Erwartung: nach deploy.sh-Änderungen einmal erneut laufen lassen.

10. **Deploy = der Test-Gate** (Retro 2026-10-05): kein CI im Repo, CI-Minuten
    sind Budget → `deploy.sh` läuft `uv run pytest` VOR dem restart (rot =
    Abbruch, Service läuft auf altem Code weiter). Vertrag bewacht von
    `scripts/lib/uniinfer-deploy-gate.test` (Metarepo, mit Self-Test).

## Typische Ein-Blick-Queries

```bash
# Welche Modelle failten heute, wie?
ssh amd 'sudo journalctl -u uniioai-proxy --since today --no-pager | grep "END POST" | grep -oE "Status: [0-9]+ - .*model=[^ ]*" | sort | uniq -c'

# SSE-Fails sichtbar machen (Status 200 + kurz + opencode):
ssh amd 'sudo journalctl -u uniioai-proxy --since "1 hour ago" --no-pager | grep "\- sse" | grep opencode'

# Was hat zen wirklich geantwortet?
ssh amd 'sudo tail -5 ~/code/python-openutils/packages/uniinfer/logs/opencode_raw_chat.log'
```
