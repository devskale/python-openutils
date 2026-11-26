"""Exit-Code-Vertrag des Modul-Aufrufs `python -m credgoo`.

Retro 2026-11-26: `credgoo/__main__.py` rief `main()` ohne `sys.exit` auf — der
Rückgabewert (0/1) verpuffte, jeder Aufrufer sah Erfolg. Gefunden über die
Skill-venvs: credgoo 0.1.14 lieferte bei „Unknown backend: cfworker" exit 0,
fetch-url/web-search fielen still auf schlüsselloses Arbeiten zurück.

Der CLI-Weg (console_script `credgoo`) war immer korrekt; dieser Test hält den
Modulweg an denselben Vertrag: Fehlercode wird weitergegeben.

Run: uv run python -m pytest tests/test_main_exit_code.py -q
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    """`python -m credgoo …` gegen ein leeres cache-dir — keine echte Konfig."""
    with tempfile.TemporaryDirectory() as tmp:
        return subprocess.run(
            [sys.executable, "-m", "credgoo", *args, "--cache-dir", tmp],
            capture_output=True, text=True, timeout=60,
        )


class ModuleExitCodeTest(unittest.TestCase):
    def test_unbekannter_dienst_liefert_exit_1(self):
        r = _run("dienst-den-es-nicht-gibt-x")
        self.assertEqual(
            r.returncode, 1,
            f"Fehler muss exit 1 melden — stdout={r.stdout!r} stderr={r.stderr!r}",
        )

    def test_version_liefert_exit_0(self):
        r = _run("--version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("credgoo", r.stdout)


if __name__ == "__main__":
    unittest.main()
