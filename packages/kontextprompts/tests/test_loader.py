"""Tests for kontextprompts.loader — 5-tier resolution + observability."""
import os
import sys
from pathlib import Path

import pytest

# allow running without install: add the package dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kontextprompts import get_prompt_set_info, load_prompt  # noqa: E402

ROUTE = "---\nversion: '1.4'\n---\nYou are a router. Pick the best doc.\n"


@pytest.fixture
def fake_clone(tmp_path, monkeypatch):
    """A fake kontext-prompts clone wired in via $KONTEXT_PROMPTS_DIR."""
    clone = tmp_path / "kontext-prompts"
    (clone / "agentos").mkdir(parents=True)
    (clone / "strukt2meta").mkdir()
    (clone / "agentos" / "routeQuery.md").write_text(ROUTE, encoding="utf-8")
    (clone / "agentos" / "extractFacts.md").write_text(
        "---\nversion: '0.2'\n---\nExtract facts.\n", encoding="utf-8"
    )
    (clone / "VERSION").write_text("commit: abc123\nbuilt_at: 2026-07-18T00:00:00Z\n")
    monkeypatch.setenv("KONTEXT_PROMPTS_DIR", str(clone))
    return clone


def test_literal_path(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("hello", encoding="utf-8")
    assert load_prompt(str(f)) == "hello"


def test_env_override_resolves(fake_clone):
    assert "router" in load_prompt("routeQuery", package="agentos")


def test_package_name_shorthand(fake_clone):
    assert "router" in load_prompt("agentos/routeQuery")


def test_strip_md_suffix(fake_clone):
    assert "router" in load_prompt("routeQuery.md", package="agentos")


def test_not_found_lists_searched(fake_clone):
    with pytest.raises(FileNotFoundError) as exc:
        load_prompt("nope", package="agentos")
    assert "KONTEXT_PROMPTS_DIR" in str(exc.value)


def test_get_prompt_set_info(fake_clone):
    info = get_prompt_set_info()
    assert info["source"] == "KONTEXT_PROMPTS_DIR"
    assert info["commit"] == "abc123"
    names = {p["name"] for p in info["prompts"]}
    assert "agentos/routeQuery.md" in names
    route = next(p for p in info["prompts"] if p["name"] == "agentos/routeQuery.md")
    assert route["version"] == "1.4"
    assert len(route["sha256"]) == 12


def test_version_none_when_no_frontmatter(fake_clone):
    (fake_clone / "strukt2meta" / "plain.md").write_text("no frontmatter here", encoding="utf-8")
    info = get_prompt_set_info()
    plain = next(p for p in info["prompts"] if p["name"] == "strukt2meta/plain.md")
    assert plain["version"] == ""


def test_fingerprint_excludes_version(fake_clone):
    """Bumping version must NOT change the semantic fingerprint."""
    from kontextprompts.loader import _semantic_fingerprint

    a = "---\nversion: '1.0'\n---\nbody\n"
    b = "---\nversion: '2.0'\n---\nbody\n"
    assert _semantic_fingerprint(a) == _semantic_fingerprint(b)
    assert _semantic_fingerprint(a) != _semantic_fingerprint("---\nversion: '1.0'\n---\nbody-changed\n")


def test_modified_drift(fake_clone):
    """modified=True only on a real semantic change vs HEAD (not a version-only bump)."""
    import subprocess

    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=fake_clone, check=True)
    subprocess.run(["git", "add", "-A"], cwd=fake_clone, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=fake_clone, check=True, env=env)

    def route():
        return next(p for p in get_prompt_set_info()["prompts"] if p["name"] == "agentos/routeQuery.md")

    # clean tree → nothing modified
    assert route()["modified"] is False

    # semantic body change → modified
    (fake_clone / "agentos" / "routeQuery.md").write_text(
        "---\nversion: '1.4'\n---\nDIFFERENT body\n", encoding="utf-8"
    )
    assert route()["modified"] is True

    # version-only change (body restored) → fingerprint matches HEAD → not modified
    (fake_clone / "agentos" / "routeQuery.md").write_text(ROUTE, encoding="utf-8")
    assert route()["modified"] is False
