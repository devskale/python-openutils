"""kontextprompts.loader — shared workflow-prompt resolver.

Single source of truth for loading the engine's *workflow* prompts (the
"how to think": retriever, router, extract*, prüfeKriterium, sheetreact, …).
Business content (PAs/FAPs) is NOT resolved here — that lives in the OFS data
layer (see ADR 0001).

5-tier resolution (first hit wins):
  1. literal file path (any extension) — read directly
  2. ``$KONTEXT_PROMPTS_DIR/<package>/<name>.md`` — explicit override
  3. default clone ``<discovered>/​<package>/​<name>.md`` — the in-place
     kontext-prompts checkout (sibling of python-utils)
  4. ``./prompts/<name>.md`` — CWD-local pack
  5. bundled in-package copy — the consumer's own ``prompts/`` (last resort)

A loud warning fires when the bundled tier is hit with no clone and no env-var,
so the invisible-drift case (running stale bundled prompts by accident) is
surfaced.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import yaml

AUTO_FIELDS = ("version", "content_sha256")  # excluded from the semantic fingerprint


# ── frontmatter + fingerprint (mirrors the kontext-prompts CLI) ──────────
def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end].lstrip("\n"), text[end + 4:].lstrip("\n")


def _semantic_fingerprint(text: str) -> str:
    """sha256 of body + frontmatter minus the auto fields (no circularity)."""
    front, body = _split_frontmatter(text)
    meta = {}
    if front.strip():
        try:
            meta = yaml.safe_load(front) or {}
        except yaml.YAMLError:
            meta = {}
    clean = {k: v for k, v in meta.items() if k not in AUTO_FIELDS}
    canonical = yaml.safe_dump(clean, sort_keys=True, allow_unicode=True).strip()
    return hashlib.sha256((canonical + "\n\u0000\n" + body).encode("utf-8")).hexdigest()


def _get_version(text: str) -> str | None:
    front, _ = _split_frontmatter(text)
    if not front.strip():
        return None
    try:
        meta = yaml.safe_load(front) or {}
    except yaml.YAMLError:
        return None
    v = meta.get("version")
    return str(v) if v is not None else None


# ── location discovery ───────────────────────────────────────────────────
def _env_path(var: str) -> Path | None:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else None


def _clone_dir() -> Path | None:
    """The default in-place kontext-prompts checkout (NOT the explicit env override)."""
    # PYTHON_UTILS_ROOT sibling: dirname($PYTHON_UTILS_ROOT)/kontext-prompts
    pur = _env_path("PYTHON_UTILS_ROOT")
    if pur:
        c = pur.parent / "kontext-prompts"
        if c.is_dir():
            return c
    # CWD sibling (dev: kontext.one/kontext-prompts/)
    c = Path.cwd() / "kontext-prompts"
    if c.is_dir():
        return c
    # pi5 / generic home layout
    c = Path.home() / "code" / "kontext-prompts"
    if c.is_dir():
        return c
    return None


def _bundled_dir(package: str | None) -> Path | None:
    """The consumer's bundled prompts/, discovered by importing the package.

    ``packages/<pkg>/<pkg>/__init__.py`` → ``packages/<pkg>/prompts``.
    """
    if not package:
        return None
    try:
        import importlib

        mod = importlib.import_module(package)
        if getattr(mod, "__file__", None):
            return Path(mod.__file__).resolve().parent.parent / "prompts"
    except Exception:
        pass
    return None


def _warn_bundled(name: str, package: str | None) -> None:
    if not _clone_dir() and not _env_path("KONTEXT_PROMPTS_DIR"):
        print(
            f"kontextprompts: WARNING — '{name}' ({package or '?'}) resolved from the "
            "BUNDLED fallback: no kontext-prompts clone found and $KONTEXT_PROMPTS_DIR "
            "unset. Clone kontext-prompts alongside python-utils or set KONTEXT_PROMPTS_DIR.",
            file=sys.stderr,
        )


# ── public API ───────────────────────────────────────────────────────────
def load_prompt(name: str, *, package: str | None = None) -> str:
    """Load a workflow prompt by name (or literal path). First matching tier wins."""
    # tier 1 — literal path
    p = Path(name).expanduser()
    if p.is_file():
        return p.read_text(encoding="utf-8")

    # "package/name" shorthand
    if "/" in name and package is None:
        head, tail = name.split("/", 1)
        if head in ("agentos", "strukt2meta"):
            package, name = head, tail

    stem = name[:-3] if name.endswith(".md") else name
    ns = package or ""

    candidates: list[tuple[str, Path]] = []
    env = _env_path("KONTEXT_PROMPTS_DIR")
    if env:
        candidates.append(("KONTEXT_PROMPTS_DIR", env / ns / f"{stem}.md"))
    clone = _clone_dir()
    if clone:
        candidates.append(("clone", clone / ns / f"{stem}.md"))
    candidates.append(("cwd", Path("prompts") / f"{stem}.md"))
    bundled = _bundled_dir(package)
    if bundled:
        candidates.append(("bundled", bundled / f"{stem}.md"))

    for label, path in candidates:
        if path.is_file():
            if label == "bundled":
                _warn_bundled(stem, package)
            return path.read_text(encoding="utf-8")

    searched = "\n  ".join(f"[{lbl}] {pth}" for lbl, pth in candidates)
    raise FileNotFoundError(
        f"workflow prompt {stem!r} (package={package or '?'}) not found. Searched:\n  {searched}"
    )


def get_prompt_set_info() -> dict:
    """Snapshot of the live prompt set: source, version, commit, per-prompt table."""
    env = _env_path("KONTEXT_PROMPTS_DIR")
    clone = _clone_dir()
    root = env or clone
    if env:
        source = "KONTEXT_PROMPTS_DIR"
    elif clone:
        source = "clone"
    else:
        source = "bundled"

    version = commit = None
    if root and (root / "VERSION").is_file():
        for line in (root / "VERSION").read_text().splitlines():
            if line.startswith("commit:"):
                commit = line.split(":", 1)[1].strip()
            if line.startswith("built_at:"):
                version = line.split(":", 1)[1].strip()

    prompts: list[dict] = []
    if root and root.is_dir():
        for ns in ("agentos", "strukt2meta"):
            ns_dir = root / ns
            if not ns_dir.is_dir():
                continue
            for md in sorted(ns_dir.rglob("*.md")):
                rel = str(md.relative_to(root))
                try:
                    text = md.read_text(encoding="utf-8")
                except OSError:
                    continue
                prompts.append(
                    {
                        "name": rel,
                        "version": _get_version(text) or "",
                        "source": source,
                        "sha256": _semantic_fingerprint(text)[:12],
                        "modified": False,
                    }
                )

    return {
        "source": source,
        "version": version,
        "commit": commit,
        "path": str(root) if root else None,
        "count": len(prompts),
        "prompts": prompts,
    }
