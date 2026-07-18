"""kontextprompts — shared workflow-prompt loader for the kontext.one engine.

Resolves workflow prompts (retriever, router, extract*, prüfeKriterium, …)
with a 5-tier precedence and reports the live prompt set. See ADR 0001.
"""
from .loader import get_prompt_set_info, load_prompt

__all__ = ["load_prompt", "get_prompt_set_info"]
__version__ = "0.1.1"
