"""Tier-qualifier stripping in the opencode docs parser — str-based (no regex)."""
from uniinfer.providers.opencode import _drop_tier_qualifier


def test_basic_tier():
    assert _drop_tier_qualifier("mimo-v2.6-flash (free)") == "mimo-v2.6-flash"

def test_no_space_before_paren():
    assert _drop_tier_qualifier("model(pro)") == "model"

def test_no_qualifier_unchanged():
    assert _drop_tier_qualifier("big-pickle") == "big-pickle"

def test_only_last_group_dropped():
    assert _drop_tier_qualifier("name (a)(b)") == "name (a)"

def test_empty_parens_dropped():
    assert _drop_tier_qualifier("name ()") == "name"

def test_unbalanced_left_alone():
    assert _drop_tier_qualifier("name (unclosed") == "name (unclosed"

def test_trailing_whitespace_stripped():
    assert _drop_tier_qualifier("name (free)   ") == "name"

def test_pathological_nested_input_last_group_still_stripped():
    # "x (a(b)" — rfind nimmt die letzte vollstaendige Gruppe "(b)" → "x (a".
    # (Das alte Regex-Verhalten strippte hier gieriger bis "x" — bewusste,
    # dokumentierte Divergenz fuer Input, der in Docs-Tabellen nicht vorkommt.)
    assert _drop_tier_qualifier("x (a(b)") == "x (a"
