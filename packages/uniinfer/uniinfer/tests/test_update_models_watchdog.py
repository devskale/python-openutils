"""POST /v1/system/update-models must never run the gather inside the proxy
cgroup where it starves the WATCHDOG pings (issue:
uniinfer-update-models-watchdog-kill). Contract:

- systemd present + polkit allows the start → 202 {"via": "systemd"}, gather
  delegated to the refresh oneshot service (own cgroup), refresh_models_file
  never touched
- polkit denies / no systemd → legacy in-process threadpool path (200/old body)
- GET /v1/system/update-models reports running + catalog age without spawning
  anything on the systemd-denied path
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uniinfer.proxy_routers.models import create_models_router
from uniinfer.proxy_services import models_registry as reg


@pytest.fixture()
def client(monkeypatch):
    app = FastAPI()
    app.include_router(create_models_router(version="test"))
    return TestClient(app)


def _fake_subprocess_run(results):
    """subprocess.run stub returning queued CompletedProcess-like objects."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return results.pop(0)

    return calls, run


class _Result:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_systemd_path_delegates_and_never_gathers(client, monkeypatch):
    monkeypatch.setattr(reg, "systemd_refresh_available", lambda: True)
    calls, run = _fake_subprocess_run([_Result(0)])
    monkeypatch.setattr(reg.subprocess, "run", run)

    async def _no_legacy(*a, **k):
        raise AssertionError("legacy gather must not run when systemd accepts")

    monkeypatch.setattr(reg, "refresh_models_file", _no_legacy)

    resp = client.post("/v1/system/update-models")
    assert resp.status_code == 202
    body = resp.json()
    assert body["via"] == "systemd"
    assert body["service"] == reg.REFRESH_SERVICE
    assert calls == [["systemctl", "start", "--no-block", reg.REFRESH_SERVICE]]


def test_polkit_denial_falls_back_to_legacy(client, monkeypatch):
    monkeypatch.setattr(reg, "systemd_refresh_available", lambda: True)
    _, run = _fake_subprocess_run([_Result(1)])
    monkeypatch.setattr(reg.subprocess, "run", run)

    async def _legacy(*a, **k):
        return {"status": "success", "message": "Models updated successfully"}

    monkeypatch.setattr(reg, "refresh_models_file", _legacy)

    resp = client.post("/v1/system/update-models")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_no_systemd_uses_legacy_without_spawning(client, monkeypatch):
    monkeypatch.setattr(reg, "systemd_refresh_available", lambda: False)

    async def _legacy(*a, **k):
        return {"status": "skipped", "message": "already running"}

    monkeypatch.setattr(reg, "refresh_models_file", _legacy)

    def _boom(cmd, **k):
        raise AssertionError("no systemctl call expected without systemd")

    monkeypatch.setattr(reg.subprocess, "run", _boom)

    resp = client.post("/v1/system/update-models")
    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"


def test_status_reports_running_and_catalog_age(client, monkeypatch):
    monkeypatch.setattr(reg, "systemd_refresh_available", lambda: True)
    _, run = _fake_subprocess_run([_Result(0, stdout="activating\n")])
    monkeypatch.setattr(reg.subprocess, "run", run)

    def _lock_locked():
        return False

    monkeypatch.setattr(reg._refresh_lock, "locked", _lock_locked)

    resp = client.get("/v1/system/update-models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is True
    assert body["service_state"] == "activating"  # oneshot gather in start-up
    assert isinstance(body["catalog_mtime"], float)  # repo ships a models.json
    assert isinstance(body["catalog_age_seconds"], float)
    assert body["catalog_age_seconds"] >= 0


def test_status_running_from_inprocess_lock(client, monkeypatch):
    monkeypatch.setattr(reg, "systemd_refresh_available", lambda: False)
    monkeypatch.setattr(
        reg._refresh_lock, "locked", lambda: True
    )

    resp = client.get("/v1/system/update-models")
    body = resp.json()
    assert body["running"] is True
    assert "service_state" not in body


def test_trigger_models_refresh_coroutine_signature():
    """trigger/status are awaitables usable from the app's event loop."""
    assert asyncio.iscoroutinefunction(reg.trigger_models_refresh)
    assert asyncio.iscoroutinefunction(reg.models_refresh_status)
