# AGENTS.md — Python Open Utils

> **Meta repo**: [kontext.one](https://github.com/devskale/kontext.one) — deploy, release, cross-repo orchestration. [`../AGENTS.md`](../AGENTS.md) ist kanonisch für Dev-Methodik; Issues-Kanban + Triage-Vokabular: [`../docs/agents/`](../docs/agents/) (`.work/issues/`, `module:`-Feld, Signatur `<machine>@<openutils>`).

## Tooling

- Python 3.9+ · **uv** (nie pip) · Monorepo: `packages/credgoo/`, `packages/uniinfer/` — je eigene `pyproject.toml` + `uv.lock`
- Setup/Tests: `cd packages/<pkg> && uv sync` · `uv run pytest [-k …]` · uniinfer-Extras: `uv sync --extra anthropic --extra gemini` (oder `--extra all`)
- Repo-Modus: Meta-Setup liegt vor, wenn `../repos.yml` existiert (Siblings = Metarepo-Layout); standalone sonst — Packages installieren via `uv sync`.

## Consumers

- python-utils importiert credgoo/uniinfer via **git URL** von GitHub (nie lokaler Pfad, nie PyPI)
- klark0 spricht den uniinfer-Proxy via `/api/ai/uniinfer/stream` (Frontend-Verbrauch: klark0 `lib/ai/` — siehe [`../klark0/AGENTS.md`](../klark0/AGENTS.md))
- External install: `uv pip install -r https://skale.dev/credgoo` / `uniinfer`

## Critical Rules

- ✅ **Always** `uv run pytest` im betroffenen Package vor dem Commit · atomare Commits mit beschreibender Message
- ⚠️ **Ask first** bei neuen Dependencies, pyproject-Änderungen, DB-Schema oder CI-Config
- 🚫 Nie Secrets/API-Keys committen · nie Plaintext-Credentials loggen (credgoo: XOR + Base64 only) · nie Dateirechte lockerer als `0o600` für gecachte Credentials

## Version Bumps

Nach Änderungen an credgoo/uniinfer:

1. Commit + push auf `main` (python-openutils)
2. Vom Metarepo: `./k.sh deps bump <pkg>` (enuminiert Locks, `-P`'t alle internen Packages + `--refresh`, Producer + alle Consumer), dann `python-utils` auf `dev` pushen

Version in `pyproject.toml`: patch `0.1.5→0.1.6` (default) / minor `0.1.5→0.2.0`. **Die komplette Kette** (inkl. dependents) macht `./k.sh deps release <pkg>` — nie hand-gerollt. uv-Cache-Interna (silent no-op, single-commit-unification, `--refresh`-Semantik) kanonisch: [`../docs/dependency-bumps.md`](../docs/dependency-bumps.md) — **vor jedem Bump lesen**.

## Deploy — zwei Ziele, nicht verwechseln

### 1. uniioai-Proxy auf `amd` (dieses Repo)

`uniioai-proxy` läuft **auf amd aus diesem Checkout** (kein git-URL-Pull — das Repo IST die Deployment-Quelle). **System-Unit, nicht `--user`**: `sudo systemctl status uniioai-proxy` · `sudo journalctl -u uniioai-proxy`. Logging-Forensik (4-Ebenen-Schema): [docs/gotchas/uniioai-proxy-logging.md](docs/gotchas/uniioai-proxy-logging.md).

**Ports/Vhosts/Timer NIEMALS hier ablesen — immer aus der Live-Config discovern** (und bei Drift DIE anpassen, nie umgekehrt):

```bash
ssh amd 'systemctl cat uniioai-proxy | grep ExecStart'        # Backend-Port (uvicorn)
ssh amd 'sudo nginx -T | grep -B3 -A8 "server_name.*uniinfer"' # öffentlicher TLS-Vhost
./scripts/lib/repos                                            # Registry: server amd in repos.yml
```

Deploy einer uniinfer-Änderung:

```bash
ssh amd 'cd code/python-openutils/packages/uniinfer && ./deploy.sh --all-extras'
# ⚠️ Immer --all-extras auf amd: bare ./deploy.sh strippt die optionalen Provider-SDKs
#    (anthropic, gemini, mistral, …) aus der venv — diese Provider brechen dann live.
```

Verwandte Units auf amd: `uniioai-proxy.service` (always-on), `uniioai-models-refresh.timer` (täglicher Model-List-Refresh).

### 2. Consumer-Apps (python-utils)

Pickup neuer Versionen läuft über die gepinnten `uv.lock`s (git URL) — Rollout auf Hosts (`deployto` → `uvinit.sh`) **nur auf explizite Anfrage**, nie automatisch.

## Package-Docs

- **UniInfer**: [`packages/uniinfer/AGENTS.md`](packages/uniinfer/AGENTS.md) — Provider-Patterns, API-Kompatibilität, async/sync · [`packages/uniinfer/CONTEXT.md`](packages/uniinfer/CONTEXT.md)
- **Credgoo**: [`packages/credgoo/AGENTS.md`](packages/credgoo/AGENTS.md) — Security, Encryption, File-Permissions · [`packages/credgoo/CONTEXT.md`](packages/credgoo/CONTEXT.md)

## Within This Repo: Local Paths

uniinfer hängt von credgoo per editable path source ab:

```toml
[tool.uv.sources]
credgoo = { path = "../credgoo", editable = true }
```

Externe Repos nutzen git URLs — Pattern: [README.md](README.md).

## Dev methodology

Kanonisch: [`../AGENTS.md`](../AGENTS.md). Hier gilt: Gates nach Lockfile-/Source-Änderungen (`cd .. && ./k.sh test scripts` + `./k.sh deps check-locks`); `git-deps`-Gate erzwingt `rev = "main"` auf internen git-Sources, `llm-gate` erzwingt llminvoke-routing; rekurrenter Gotcha → hier; objektiv prüfbar → `*.test`-Gate im Metarepo.

## Code Style

- **No comments** unless explicitly requested · **Docstrings** für alle öffentlichen Funktionen/Klassen/Module · **Type hints** wo passend · Imports: stdlib → third-party → local (isort)
