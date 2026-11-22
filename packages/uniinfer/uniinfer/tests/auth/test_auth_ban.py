"""
Proxy auth-ban: after N failed auth attempts an IP is blocked for Y hours.

In-process fail2ban for the uniioai proxy (LeanHTTPMiddleware). Client IPs
come from the socket peer; behind nginx (loopback peer) the RIGHTMOST
X-Forwarded-For entry is trusted — our own proxy appends $remote_addr there,
so leftmost entries are attacker-spoofable and ignored. Loopback is never
banned (own smoke tests must not lock themselves out).
"""
import asyncio
from unittest.mock import patch

import pytest

import uniinfer.proxy_middleware as pm


def _bans(threshold=10, hours=2):
    b = pm._AuthBan()
    b.threshold = threshold
    b.hours = hours
    return b


async def run_req(mw, scope_client, status=401, xff=None):
    headers = [(b"content-length", b"2")]
    if xff:
        headers.append((b"x-forwarded-for", xff))
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "client": (scope_client, 1234), "headers": headers}
    sent = []
    calls = {"n": 0}

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        sent.append(msg)

    async def app(scope, receive, send):
        calls["n"] += 1
        await send({"type": "http.response.start", "status": status,
                    "headers": [("content-type", "application/json")]})
        await send({"type": "http.response.body", "body": b"{}"})

    mw.app = app
    await mw(scope, receive, send)
    return sent[0]["status"], calls["n"]


class TestClientIP:
    def test_loopback_peer_uses_rightmost_xff(self):
        ip, trusted = pm._client_ip({"client": ("127.0.0.1", 5),
                                     "headers": [(b"x-forwarded-for", b"127.0.0.1, 23.27.48.61")]})
        assert ip == "23.27.48.61"
        assert trusted is True

    def test_external_peer_ignores_spoofed_xff(self):
        ip, trusted = pm._client_ip({"client": ("23.27.48.61", 5),
                                     "headers": [(b"x-forwarded-for", b"127.0.0.1")]})
        assert ip == "23.27.48.61"
        assert trusted is False

    def test_no_xff_falls_back_to_peer(self):
        ip, _ = pm._client_ip({"client": ("194.96.252.52", 5), "headers": []})
        assert ip == "194.96.252.52"


class TestAuthBan:
    @pytest.mark.asyncio
    async def test_ban_after_threshold_blocks_without_app_call(self):
        bans = _bans(threshold=10)
        mw = pm.LeanHTTPMiddleware(None)
        with patch.object(pm, "_AUTH_BANS", bans):
            statuses = []
            for i in range(9):
                st, calls = await run_req(mw, "23.27.48.61", 401, xff=b"23.27.48.61")
                statuses.append((st, calls))
            # Unter dem Schwellwert: echte 401s, App läuft.
            assert all(st == 401 and c == 1 for st, c in statuses)

            st, calls = await run_req(mw, "23.27.48.61", 401, xff=b"23.27.48.61")
            assert st == 401 and calls == 1  # 10. Fail wird gezählt und noch bedient...

            st, calls = await run_req(mw, "23.27.48.61", 200, xff=b"23.27.48.61")
            assert st == 429 and calls == 0  # …danach: geblockt ohne App-Aufruf (auch gültige Keys)

    @pytest.mark.asyncio
    async def test_ips_are_independent_and_200_never_counts(self):
        bans = _bans(threshold=3)
        mw = pm.LeanHTTPMiddleware(None)
        with patch.object(pm, "_AUTH_BANS", bans):
            for i in range(2):
                await run_req(mw, "1.1.1.1", 401)
            await run_req(mw, "1.1.1.1", 200)  # Erfolg zählt nicht als Fail
            st, calls = await run_req(mw, "1.1.1.1", 401)
            assert st == 401 and calls == 1    # 3. Fail bannt (noch bedient)…
            st, calls = await run_req(mw, "1.1.1.1", 401)
            assert st == 429 and calls == 0    # …ab dem nächsten Request blockiert

            st, calls = await run_req(mw, "2.2.2.2", 401)
            assert st == 401 and calls == 1    # andere IP völlig unabhängig

    @pytest.mark.asyncio
    async def test_success_stops_failure_accumulation_but_old_fails_remain(self):
        bans = _bans(threshold=3)
        mw = pm.LeanHTTPMiddleware(None)
        with patch.object(pm, "_AUTH_BANS", bans):
            await run_req(mw, "3.3.3.3", 401)
            await run_req(mw, "3.3.3.3", 200)
            st, _ = await run_req(mw, "3.3.3.3", 401)  # 2. Fail innerhalb Window
            assert st == 401

    @pytest.mark.asyncio
    async def test_loopback_is_never_banned(self):
        bans = _bans(threshold=3)
        mw = pm.LeanHTTPMiddleware(None)
        with patch.object(pm, "_AUTH_BANS", bans):
            for i in range(12):
                st, calls = await run_req(mw, "127.0.0.1", 401)
            assert st != 429 and calls == 1

    def test_remaining_and_expiry_semantics(self):
        bans = _bans(hours=2)
        t0 = 1_000_000.0
        with patch.object(pm.time, "time", lambda: t0):
            assert bans.remaining("x") == 0
            with patch.object(bans, "record_failure"):  # ban direkt setzen
                pass
            bans.banned_until["x"] = t0 + 3600
            assert 3599 < bans.remaining("x") <= 3600
