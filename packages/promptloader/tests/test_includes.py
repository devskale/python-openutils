"""Include-Support ({{> pfad.md}}) — Semantik-Vertrag.

Eine Quelle für geteilte Regelblöcke: Prompts betten Per Partial ein, der
Loader löst beim LADEN auf (die LLm sieht nur aufgelösten Text). Der
kontext-prompts-CLI spiegelt die Syntax fürs Version-Bumping — diese Tests
sind der gemeinsame Bezugspunkt (Syntax hier = Syntax dort).
"""

from pathlib import Path

import pytest

from promptloader.loader import resolve_includes


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    (tmp_path / "_regeln").mkdir()
    (tmp_path / "_regeln" / "datum-format.md").write_text(
        "---\nversion: '1.0'\n---\n- Daten im Fließtext IMMER als TT.MM.JJJJ.\n",
        encoding="utf-8",
    )
    (tmp_path / "_regeln" / "a.md").write_text("A\n{{> _regeln/b.md}}\n", encoding="utf-8")
    (tmp_path / "_regeln" / "b.md").write_text("B-Inhalt\n", encoding="utf-8")
    return tmp_path


def test_include_wird_aufgeloest(tree):
    out = resolve_includes("Regeln:\n{{> _regeln/datum-format.md}}\nEnde.", tree)
    assert "TT.MM.JJJJ" in out and "{{>" not in out


def test_partial_frontmatter_wird_gestripppt(tree):
    out = resolve_includes("{{> _regeln/datum-format.md}}", tree)
    assert "version" not in out and "---" not in out


def test_fehlendes_partial_fail_loud(tree):
    with pytest.raises(FileNotFoundError, match="gibtsnicht.md"):
        resolve_includes("{{> _regeln/gibtsnicht.md}}", tree)


def test_zyklus_fail_loud(tree):
    # b.md in a.md inkludieren lassen -> Zyklus a -> b -> a
    (tree / "_regeln" / "b.md").write_text("B\n{{> _regeln/a.md}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Zyklus"):
        resolve_includes("{{> _regeln/a.md}}", tree)


def test_tiefe_begrenzt(tree):
    chain = "start"
    for i in range(8):
        nxt = f"_regeln/k{i}.md"
        (tree / nxt).write_text(f"k{i}\n{{{{> _regeln/k{i+1}.md}}}}\n", encoding="utf-8")
    (tree / "_regeln" / "k8.md").write_text(" ende\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Tiefe"):
        resolve_includes("{{> _regeln/k0.md}}", tree)


def test_ohne_include_unchanged(tree):
    assert resolve_includes("ganz normaler text", tree) == "ganz normaler text"
