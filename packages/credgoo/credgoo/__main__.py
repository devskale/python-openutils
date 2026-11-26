import sys

from .credgoo import main

# Exit-Code-Vertrag wie beim CLI (console_script → sys.exit(main())): ohne
# sys.exit verpufft der Rückgabewert von main() und jeder Fehler sieht wie
# Erfolg aus (Retro 2026-11-26: credgoo 0.1.14 in Skill-venvs meldete
# „Unknown backend" mit exit 0). Test: tests/test_main_exit_code.py
sys.exit(main())
