"""Decision-Takt (klark0-systemone-retry-storm, 2026-11-21).

Vorfall: tu-robotni-Worker-Batch-Parallelität feuerte >4 req/s /v1/systemone
gegen ein 240/min-Gateway-Bucket (hab-Chat proxy-weit erdrückt). Der Takt:
prozessweiter Mindestabstand zwischen Decisions + gemeinsamer 429-Cooldown.

Hermetisch: keine Netz-Calls — _paced/_pace_note_rate_limit sind Pure-Timing
mit injizierten Intervallen; die decide()-429-Verdrahtung wird über einen
gemockten Transport geprüft.
"""
import threading
import time
import urllib.error
import pytest

import llminvoke as li


def test_pace_min_interval(monkeypatch):
    monkeypatch.setattr(li, "_DECISION_MIN_INTERVAL", 0.08)
    li._decision_pace_state.update(last=0.0, cooldown_until=0.0)
    t0 = time.monotonic()
    li._paced()
    li._paced()  # zweiter Call muss den Takt halten
    dt = time.monotonic() - t0
    assert dt >= 0.08, f"Min-Intervall nicht durchgehalten: {dt:.3f}s"


def test_pace_parallel_threads_share_takt(monkeypatch):
    monkeypatch.setattr(li, "_DECISION_MIN_INTERVAL", 0.05)
    li._decision_pace_state.update(last=0.0, cooldown_until=0.0)
    hits = []
    lock = threading.Lock()

    def worker():
        li._paced()
        with lock:
            hits.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hits.sort()
    # je zwei aufeinanderfolgende Decisions ≥ min-intervall (gleicher Prozess)
    for a, b in zip(hits, hits[1:]):
        assert b - a >= 0.05 - 0.02, f"Threads liefen ungetaktet durch: {b - a:.3f}s"


def test_pace_cooldown_after_rate_limit(monkeypatch):
    monkeypatch.setattr(li, "_DECISION_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(li, "_DECISION_429_COOLDOWN", 0.1)
    li._decision_pace_state.update(last=0.0, cooldown_until=0.0)
    li._pace_note_rate_limit()
    t0 = time.monotonic()
    li._paced()  # muss bis zum Cooldown-Ablauf warten
    assert time.monotonic() - t0 >= 0.08, "429-Cooldown nicht durchgehalten"


def test_decide_notes_rate_limit_on_429(monkeypatch):
    """Die Retry-Schleife setzt den gemeinsamen Cooldown bei einer 429."""
    monkeypatch.setattr(li, "_DECISION_MIN_INTERVAL", 0.0)

    class FakeResp:
        def read(self):
            return b"{}"

        def headers(self):
            return {}

    def fake_post(url, bearer, payload, timeout=45.0):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, FakeResp())  # type: ignore[arg-type]

    monkeypatch.setattr(li, "_systemone_post", fake_post)
    li._decision_pace_state.update(last=0.0, cooldown_until=0.0)
    with pytest.raises(Exception):
        li.decide(
            state={"text": "x"},
            questions=[{"id": "q", "type": "choice", "question": "ok?", "labels": ["a", "b"]}],
            model="hab@LiquidAI/d1-2B",
            max_attempts=2,
        )
    assert li._decision_pace_state["cooldown_until"] > time.monotonic(), "429 hat keinen Cooldown gesetzt"


def test_pace_box_global_across_processes(monkeypatch, tmp_path):
    """Subprozesse teilen den Takt über die Datei (Vorfall: 251/min trotz
    Prozess-Takt — pdf2md-Jobs sind je eigene Prozesse)."""
    import subprocess
    import sys

    pace_file = tmp_path / "pace.json"
    monkeypatch.setenv("LLMINVOKE_DECISION_PACE_FILE", str(pace_file))
    monkeypatch.setattr(li, "_DECISION_MIN_INTERVAL", 0.12)
    code = (
        "import time, llminvoke as li; "
        "li._decision_pace_state.update(last=0.0, cooldown_until=0.0); "
        "li._paced(); print(time.time())"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(3)]
    stamps = []
    for p in procs:
        out, _ = p.communicate(timeout=15)
        stamps.append(float(out.strip()))
    stamps.sort()
    assert stamps[-1] - stamps[0] >= 0.12 - 0.04, f"Prozesse liefen ungetaktet: {stamps}"
