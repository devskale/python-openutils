"""kontextprompts — shared workflow-prompt loader for the kontext.one engine.

Resolves workflow prompts (retriever, router, extract*, prüfeKriterium, …)
with a 5-tier precedence and reports the live prompt set. See ADR 0001.
"""
from .loader import (
    frontmatter_version,
    get_prompt_set_info,
    get_prompts_root,
    load_prompt,
    split_frontmatter,
)

__all__ = [
    "load_prompt",
    "get_prompt_set_info",
    "get_prompts_root",
    "split_frontmatter",  # R7 (robotni-Review): öffentliche Fläche statt _-Importe
    "frontmatter_version",
]
# Single Source: pyproject ist die einzige Versionsquelle — hier nur lesen
# (Hardcode-Duplikat driftete: Bump erreichte pyproject, nicht __version__)
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("promptloader")
except PackageNotFoundError:  # Quell-Checkout ohne Installation
    __version__ = "0.0.0+source"
