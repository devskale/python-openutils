# kontextprompts

Shared workflow-prompt loader for the **kontext.one** engine. Resolves the
engine's *how to think* prompts (retriever, router, extract*, `prüfeKriterium`,
sheetreact, …) with a 5-tier precedence, and reports the live prompt set for
observability. Business content (PAs/FAPs) is **not** resolved here — that lives
in the OFS data layer (ADR 0001).

## Install

```bash
pip install git+https://github.com/devskale/python-openutils.git#subdirectory=packages/kontextprompts
```

In a `uv` workspace, add to `[tool.uv.sources]`:
```toml
kontextprompts = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/kontextprompts" }
```

## Resolution (first hit wins)

`load_prompt(name, *, package=None)`:

1. **literal path** — `name` is an existing file → read directly
2. **`$KONTEXT_PROMPTS_DIR`** → `$KONTEXT_PROMPTS_DIR/<package>/<name>.md` (explicit override)
3. **default clone** → the in-place `kontext-prompts` checkout, discovered as:
   `dirname($PYTHON_UTILS_ROOT)/kontext-prompts`, then `./kontext-prompts`, then `~/code/kontext-prompts`
4. **`./prompts/<name>.md`** — CWD-local pack
5. **bundled** — the consumer's own `prompts/` (discovered by importing `package`); last resort

A **loud warning** fires when the bundled tier is hit with no clone and no
env-var — the invisible-drift case (running stale bundled prompts by accident).

`"agentos/foo"` is accepted as shorthand for `load_prompt("foo", package="agentos")`.

## Observability

```python
from kontextprompts import get_prompt_set_info
get_prompt_set_info()
# → {source: "clone", version, commit, path, count, prompts: [{name, version, source, sha256, modified}]}
```

`source` ∈ `clone` | `KONTEXT_PROMPTS_DIR` | `bundled`. The `prompts` block feeds
the worker `/api/worker/version` endpoint and the klark0 status page.
