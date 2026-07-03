# AGENTS.md — Python Open Utils

> **Meta repo**: [kontext.one](https://github.com/devskale/kontext.one) — deploy, release, cross-repo orchestration. See [GUIDE.md](https://github.com/devskale/kontext.one/blob/main/GUIDE.md).

## Tooling

- Python 3.9+, **uv** for package management (not pip)
- Monorepo: `packages/credgoo/` and `packages/uniinfer/` — each has own `pyproject.toml` + `uv.lock`

## Commands

```bash
# Per-package sync (creates venv, installs from lockfile)
cd packages/<pkg> && uv sync

# Tests
cd packages/<pkg> && uv run pytest
cd packages/<pkg> && uv run pytest -k "test_name"          # single test

# Optional provider extras (uniinfer only)
cd packages/uniinfer && uv sync --extra anthropic --extra gemini
cd packages/uniinfer && uv sync --extra all
```

## Repo Modes

**Meta setup** (`../repos.yml` exists):

| Repo | Path |
|------|------|
| kontext.one (meta) | `..` |
| klark0 | `../klark0` |
| python-utils | `../python-utils` |

**Standalone**: no siblings. Packages install via `uv sync`.

## Consumers

- python-utils imports credgoo/uniinfer via **git URL** from GitHub (not local path)
- klark0 calls uniinfer proxy via `/api/ai/uniinfer/stream` — see `../klark0/lib/ai/uniinferApi.ts`
- External install: `uv pip install -r https://skale.dev/credgoo` / `uniinfer`

## Frontend Code

| Backend | Frontend (klark0) | GitHub |
|---------|-------------------|--------|
| Uniinfer streaming | `../klark0/app/api/ai/uniinfer/stream/route.ts` | [route.ts](https://github.com/devskale/klark0/blob/main/app/api/ai/uniinfer/stream/route.ts) |
| Uniinfer client | `../klark0/lib/ai/uniinferApi.ts` | [uniinferApi.ts](https://github.com/devskale/klark0/blob/main/lib/ai/uniinferApi.ts) |
| AI settings | `../klark0/lib/ai/settings.ts` | [settings.ts](https://github.com/devskale/klark0/blob/main/lib/ai/settings.ts) |

## Critical Rules

- ✅ **Always** run `uv run pytest` in the relevant package before committing
- ✅ **Always** make atomic commits with descriptive messages
- ⚠️ **Ask first** before adding dependencies or changing pyproject.toml
- ⚠️ **Ask first** before modifying database schema or CI config
- 🚫 **Never** commit secrets or API keys
- 🚫 **Never** log plaintext credentials (credgoo: XOR + Base64 only)
- 🚫 **Never** use file permissions looser than `0o600` for cached credentials

## Within This Repo: Local Paths

uniinfer depends on credgoo via editable path source:
```toml
[tool.uv.sources]
credgoo = { path = "../credgoo", editable = true }
```

External repos use git URLs — see README.md for the pattern.

## Version Bumps

After changing credgoo or uniinfer:
1. Commit and push to `main` on python-openutils
2. In python-utils: `cd packages/THAT_PACKAGE && uv lock -U`, then push to `dev`

Version in `pyproject.toml`:
- **Patch** (default): `0.1.5` → `0.1.6`
- **Minor**: `0.1.5` → `0.2.0`

## Deploy

There are two deploy targets — do not confuse them.

### 1. The uniioai proxy on `amd` (this host)

The `uniioai-proxy` systemd service runs **on amd** from this local checkout
(`/home/ubuntu/code/python-openutils/packages/uniinfer`), served by uvicorn on
port `8124` (nginx TLS front on `8123`). It is **not** pulled via git URL —
it *is* this repo.

To deploy a uniinfer change to the live proxy:
```bash
ssh amd 'cd code/python-openutils/packages/uniinfer && ./deploy.sh'
# deploy.sh = git pull → uv sync --all-extras → sudo systemctl restart uniioai-proxy
```
If you're already on amd editing in-place, the equivalent is:
```bash
cd packages/uniinfer && uv sync --all-extras && sudo systemctl restart uniioai-proxy
```
Related systemd units on amd: `uniioai-proxy.service` (always-on),
`uniioai-models-refresh.timer` (daily model-list refresh at 04:00 UTC).

### 2. Consumer apps (python-utils)

`python-utils` packages import uniinfer via **git URL** from GitHub, so they
pick up new versions through their pinned `uv.lock` after step 2 above.

Rolling those changes out to the hosts that run those apps
(`deployto` → `uvinit.sh` → `uv sync`) is **only done on explicit request** —
do not run it automatically.

## Package-Specific Docs

- **UniInfer**: `packages/uniinfer/AGENTS.md` — provider patterns, API compatibility, async/sync
- **Credgoo**: `packages/credgoo/AGENTS.md` — security, encryption, file permissions

## Code Style

- **No comments** unless explicitly requested
- **Docstrings** required for all public functions/classes/modules
- **Type hints** where appropriate
- Imports: stdlib → third-party → local (use `isort`)
