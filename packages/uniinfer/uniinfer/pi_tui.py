"""Curses-based hierarchical checkbox selector for picking catalog models.

No third-party dependency — uses the stdlib ``curses`` module (macOS/Linux).
Falls back to a plain number-based prompt when stdin is not a TTY (or curses
is unavailable, e.g. Windows).

UX:
    ↑/↓ or k/j   navigate
    space        toggle (provider row = all its models; model row = that model)
    →/←          expand/collapse a provider
    a / n        select all / none
    enter        confirm
    q / esc      cancel
"""
from __future__ import annotations

import curses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Row:
    kind: str  # "provider" | "model"
    provider: str
    model_id: str = ""
    model: Optional[dict] = None
    expanded: bool = False


def _group(pairs: list[tuple[str, dict]]) -> tuple[list[_Row], dict[str, list[int]]]:
    """Build provider rows + index of their model-row positions in the display."""
    rows: list[_Row] = []
    order: dict[str, list[int]] = {}
    seen_provs: dict[str, int] = {}
    for prov, m in pairs:
        if prov not in seen_provs:
            seen_provs[prov] = len(rows)
            rows.append(_Row(kind="provider", provider=prov, expanded=True))
            order[prov] = []
        rows.append(_Row(kind="model", provider=prov, model_id=m.get("id", ""),
                         model=m))
        order[prov].append(len(rows) - 1)
    return rows, order


def _display_rows(rows: list[_Row]) -> list[int]:
    """Indices into ``rows`` that are currently visible (provider always; models
    only when their provider is expanded)."""
    out = []
    show_models = False
    for i, r in enumerate(rows):
        if r.kind == "provider":
            out.append(i)
            show_models = r.expanded
        elif show_models:
            out.append(i)
    return out


def _provider_state(rows, order, sel, prov):
    """'all', 'some', or 'none' for a provider's selection."""
    chosen = sum(1 for i in order[prov] if (rows[i].provider, rows[i].model_id) in sel)
    total = len(order[prov])
    if chosen == 0:
        return "none", chosen, total
    return ("all" if chosen == total else "some"), chosen, total


def _tag_str(m: dict) -> str:
    mods = (m.get("modalities") or {}).get("input") or []
    caps = m.get("capabilities") or {}
    tags = []
    if "image" in mods or caps.get("vision"):
        tags.append("img")
    if caps.get("reasoning"):
        tags.append("reason")
    ctx = m.get("context_window")
    if isinstance(ctx, int):
        tags.append(f"{ctx // 1000}K")
    return " ".join(tags)


def _run(stdscr, rows, order, title):
    sel: set[tuple[str, str]] = set()
    visible = _display_rows(rows)
    cursor = 0  # index into `visible`
    top = 0
    stdscr.keypad(True)
    curses.curs_set(0)

    while True:
        h, w = stdscr.getmaxyx()
        # clamp scroll so the cursor stays visible
        if cursor < top:
            top = cursor
        if cursor >= top + h - 2:
            top = cursor - h + 3
        if top < 0:
            top = 0

        stdscr.erase()
        stdscr.addstr(0, 0, title[: w - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, "↑↓ move · space toggle · →← expand · a all · n none · enter confirm · q cancel"[: w - 1], curses.A_DIM)

        for vi, ri in enumerate(visible):
            if vi < top or vi >= top + h - 2:
                continue
            r = rows[ri]
            y = vi - top + 2
            mark = " "
            text = ""
            if r.kind == "provider":
                state, chosen, total = _provider_state(rows, order, sel, r.provider)
                mark = {"all": "✓", "some": "~", "none": " "}[state]
                flag = "▾" if r.expanded else "▸"
                text = f"{flag} [{mark}] {r.provider} ({chosen}/{total})"
            else:
                on = (r.provider, r.model_id) in sel
                mark = "✓" if on else " "
                tags = _tag_str(r.model or {})
                text = f"    [{mark}] {r.model_id}"
                if tags:
                    text += f"  {tags}"
            attr = curses.A_REVERSE if vi == cursor else curses.A_NORMAL
            try:
                stdscr.addstr(y, 0, text[: w - 1], attr)
            except curses.error:
                pass
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(visible)
        elif ch in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(visible)
        elif ch in (ord(" "),):
            r = rows[visible[cursor]]
            if r.kind == "provider":
                state, _, _ = _provider_state(rows, order, sel, r.provider)
                if state == "all":
                    for i in order[r.provider]:
                        sel.discard((rows[i].provider, rows[i].model_id))
                else:
                    for i in order[r.provider]:
                        sel.add((rows[i].provider, rows[i].model_id))
            else:
                key = (r.provider, r.model_id)
                if key in sel:
                    sel.discard(key)
                else:
                    sel.add(key)
        elif ch in (curses.KEY_RIGHT, ord("l")):
            r = rows[visible[cursor]]
            if r.kind == "provider":
                r.expanded = True
                visible = _display_rows(rows)
        elif ch in (curses.KEY_LEFT, ord("h")):
            r = rows[visible[cursor]]
            if r.kind == "provider":
                r.expanded = False
                visible = _display_rows(rows)
                # clamp cursor
                cursor = min(cursor, len(visible) - 1)
        elif ch == ord("a"):
            for r in rows:
                if r.kind == "model":
                    sel.add((r.provider, r.model_id))
        elif ch == ord("n"):
            sel.clear()
        elif ch in (ord("q"), 27):
            return None
        elif ch in (10, 13, curses.KEY_ENTER):
            return sel


def interactive_pick_models(
    pairs: list[tuple[str, dict]], title: str = "Select accessible models"
) -> list[tuple[str, dict]] | None:
    """Run a curses checkbox selector over (provider, model) pairs.

    Returns the selected pairs, or None if cancelled / unavailable.
    """
    rows, order = _group(pairs)
    if not rows:
        return []
    # Let exceptions propagate so the caller can fall back to a non-TUI prompt.
    sel = curses.wrapper(_run, rows, order, title)
    if sel is None:
        return None
    return [(p, m) for (p, m) in pairs if (p, m.get("id", "")) in sel]
