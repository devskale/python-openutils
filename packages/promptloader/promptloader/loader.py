"""kontextprompts.loader — shared workflow-prompt resolver.

Single source of truth for loading the engine's *workflow* prompts (the
"how to think": retriever, router, extract*, prüfeKriterium, sheetreact, …).
Business content (PAs/FAPs) is NOT resolved here — that lives in the OFS data
layer (see ADR 0001).

Resolution (first matching tier wins):
  1. literal file path (any extension) — read directly
  2. ``$KONTEXT_PROMPTS_DIR/<package>/`` — explicit override tree
  3. default clone ``<discovered>/<package>/`` — the in-place kontext-prompts checkout
  4. ``./prompts/`` — CWD-local pack
  5. bundled in-package copy — the consumer's own ``prompts/`` (last resort)

Within a package tree the loader resolves by **basename** recursively (prompts
may live in functional subdirs — ADR 0002); ``README.md``, anything under a
``_*``-prefixed or ``docs/`` dir is excluded from the active set. Basenames must
be unique within a package (the index build raises on a duplicate). A loud
warning fires when the bundled tier is hit with no clone and no env-var.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
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


def _resolve_pkg_dir(root: Path, package: str) -> Path | None:
    """Find a package directory — checks system/ first (new structure), then root (legacy)."""
    for candidate in (root / "system" / package, root / package):
        if candidate.is_dir():
            return candidate
    return None
    v = os.environ.get(var)
    return Path(v).expanduser() if v else None


def _clone_dir() -> Path | None:
    """The default in-place kontext-prompts checkout (NOT the explicit env override)."""
    pur = _env_path("PYTHON_UTILS_ROOT")
    if pur:
        c = pur.parent / "kontext-prompts"
        if c.is_dir():
            return c
    c = Path.cwd() / "kontext-prompts"
    if c.is_dir():
        return c
    c = Path.home() / "code" / "kontext-prompts"
    if c.is_dir():
        return c
    return None


def get_prompts_root() -> Path | None:
    """The resolved kontext-prompts root directory (clone or env override).

    Use this to access non-prompt content (e.g. ``fachbibliothek/pas/``,
    ``fachbibliothek/faps/``) that lives in the repo but isn't loaded by
    :func:`load_prompt`.
    """
    env = _env_path("KONTEXT_PROMPTS_DIR")
    if env and env.is_dir():
        return env
    return _clone_dir()


def _bundled_dir(package: str | None) -> Path | None:
    """The consumer's bundled prompts/, discovered by importing the package."""
    if not package:
        return None
    try:
        import importlib

        mod = importlib.import_module(package)
        bases = []
        if getattr(mod, "__path__", None):
            bases = [Path(p).resolve() for p in mod.__path__]
        elif getattr(mod, "__file__", None):
            bases = [Path(mod.__file__).resolve().parent]
        for base in bases:
            for cand in (base.parent / "prompts", base / "prompts"):
                if cand.is_dir():
                    return cand
    except Exception:
        pass
    return None


def _git_head_text(root: Path, rel: str) -> str | None:
    """File contents at HEAD inside `root` (a git repo), or None if absent/not a repo."""
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], cwd=str(root), capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else None


def _warn_bundled(name: str, package: str | None) -> None:
    if not _clone_dir() and not _env_path("KONTEXT_PROMPTS_DIR"):
        print(
            f"kontextprompts: WARNING — '{name}' ({package or '?'}) resolved from the "
            "BUNDLED fallback: no kontext-prompts clone found and $KONTEXT_PROMPTS_DIR "
            "unset. Clone kontext-prompts alongside python-utils or set KONTEXT_PROMPTS_DIR.",
            file=sys.stderr,
        )


# ── recursive active-set index (ADR 0002) ────────────────────────────────
_INDEX_CACHE: dict[str, dict[str, Path]] = {}


def _is_excluded(path: Path, pkg_dir: Path) -> bool:
    """Skip README.md and anything under a _*-prefixed or `docs/` dir (within pkg_dir)."""
    if path.name.lower() == "readme.md":
        return True
    try:
        ancestors = path.relative_to(pkg_dir).parts[:-1]
    except ValueError:
        return False
    return any(p == "docs" or p.startswith("_") for p in ancestors)


def _index(pkg_dir: Path) -> dict[str, Path]:
    """Cached basename→path map of active prompts under pkg_dir (recursive, skip rules).

    Enforces per-package basename uniqueness — raises FileNotFoundError on a duplicate.
    """
    key = str(pkg_dir)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    idx: dict[str, Path] = {}
    for path in sorted(pkg_dir.rglob("*.md")):
        if _is_excluded(path, pkg_dir):
            continue
        base = path.stem
        if base in idx:
            raise FileNotFoundError(
                f"prompt name {base!r} is not unique under {pkg_dir}: "
                f"{idx[base]} and {path}"
            )
        idx[base] = path
    _INDEX_CACHE[key] = idx
    return idx


def _resolve_in_dir(pkg_dir: Path, stem: str) -> Path | None:
    """Resolve <stem>.md anywhere under pkg_dir (recursive, cached). None if absent/not a dir."""
    if not pkg_dir.is_dir():
        return None
    return _index(pkg_dir).get(stem)


def _active_prompts(pkg_dir: Path):
    """Yield active prompt paths under pkg_dir (recursive, skip rules), sorted."""
    if not pkg_dir.is_dir():
        return []
    return [p for p in sorted(pkg_dir.rglob("*.md")) if not _is_excluded(p, pkg_dir)]


# ── public API ───────────────────────────────────────────────────────────
def load_prompt(name: str, *, package: str | None = None) -> str:
    """Load a workflow prompt by basename (or literal path). First matching tier wins.

    Within a package tree the basename is resolved recursively (prompts may live in
    functional subdirs); ``README.md``/``_*``/``docs`` are excluded.
    """
    # tier 1 — literal path
    p = Path(name).expanduser()
    if p.is_file():
        return p.read_text(encoding="utf-8")

    # "package/name" shorthand
    if "/" in name and package is None:
        head, tail = name.split("/", 1)
        if head in ("agentos", "strukt2meta", "pdf2md"):
            package, name = head, tail

    stem = name[:-3] if name.endswith(".md") else name
    ns = package or ""
    env = _env_path("KONTEXT_PROMPTS_DIR")
    clone = _clone_dir()
    bundled = _bundled_dir(package)

    # ordered tiers — each resolves <stem>.md anywhere under its package tree
    tiers: list[tuple[str, Path | None]] = [
        ("KONTEXT_PROMPTS_DIR", _resolve_pkg_dir(env, ns) if env else None),
        ("clone", _resolve_pkg_dir(clone, ns) if clone else None),
        ("cwd", Path("prompts")),
        ("bundled", bundled),
    ]
    searched = []
    for label, pkg_dir in tiers:
        if pkg_dir is None:
            continue
        path = _resolve_in_dir(pkg_dir, stem)
        searched.append(f"[{label}] {pkg_dir}")
        if path is not None:
            if label == "bundled":
                _warn_bundled(stem, package)
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"workflow prompt {stem!r} (package={package or '?'}) not found. Searched:\n  "
        + "\n  ".join(searched)
    )


def get_prompt_set_info() -> dict:
    """Snapshot of the live prompt set: source, version, commit, per-prompt table."""
    env = _env_path("KONTEXT_PROMPTS_DIR")
    clone = _clone_dir()
    root = env or clone
    source = "KONTEXT_PROMPTS_DIR" if env else ("clone" if clone else "bundled")

    version = commit = None
    if root and (root / "VERSION").is_file():
        for line in (root / "VERSION").read_text().splitlines():
            if line.startswith("commit:"):
                commit = line.split(":", 1)[1].strip()
            if line.startswith("built_at:"):
                version = line.split(":", 1)[1].strip()

    prompts: list[dict] = []
    if root and root.is_dir():
        for ns in ("agentos", "strukt2meta", "pdf2md"):
            for md in _active_prompts(root / ns):
                rel = str(md.relative_to(root))
                try:
                    text = md.read_text(encoding="utf-8")
                except OSError:
                    continue
                head = _git_head_text(root, rel)
                prompts.append(
                    {
                        "name": rel,
                        "version": _get_version(text) or "",
                        "source": source,
                        "sha256": _semantic_fingerprint(text)[:12],
                        "modified": head is not None
                        and _semantic_fingerprint(text) != _semantic_fingerprint(head),
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
