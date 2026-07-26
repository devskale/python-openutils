# Issue tracker → cross-machine kanban

Issues for this repo live in the **shared kontext.one kanban**
(`.handoff/kontext.one/issues/`), **not GitHub Issues** — one pool across the
whole ecosystem, discriminated by the `module:` field (`credgoo`, `uniinfer`).

**Canonical reference (read first):**
[`kontext.one/.pi/skills/issues/SKILL.md`](https://github.com/devskale/kontext.one/blob/main/.pi/skills/issues/SKILL.md)
— file format, state flow, full command list, identity model.

Quick start (from anywhere in this repo):

```bash
../.pi/scripts/issues board           # overview (backlog/active/review/...)
../.pi/scripts/issues ls active       # in progress
../.pi/scripts/issues todo            # waiting on THIS machine
../.pi/scripts/issues new <slug> [to] # create in backlog
```

Sign edits with `<machine>@<module>` (machine from `~/.handoff-me`).
