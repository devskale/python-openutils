"""Simple tests for the adaptive AIMD rate limiter.

A ``FakeProvider`` enforces a sliding-window requests/minute cap (changeable
live to emulate a provider upgrade). The limiter drives request timing against
it via an injectable fake clock + fake sleep (no real waiting, no global
patching). Tests prove the core claims directly:

* a 429 lowers the estimated rate;
* a success raises it (once stable);
* adaptation keeps 429s rare where a fixed high rate does not;
* the limiter re-discovers a higher limit after an upgrade;
* learned limits persist across a reload.
"""
from datetime import datetime, timedelta

import pytest

from uniinfer.ratelimit import AdaptiveRateLimiter, all_rate_limiter_status, get_rate_limiter


class FakeClock:
    def __init__(self, start):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += timedelta(seconds=seconds)


class FakeProvider:
    """Backend that 200/429s against a sliding ``limit_rpm`` per-minute cap."""

    def __init__(self, clock, limit_rpm, window=60.0):
        self.clock = clock
        self.limit_rpm = limit_rpm
        self.window = window
        self._attempts = []
        self.n429 = 0

    def set_limit(self, rpm):
        self.limit_rpm = rpm

    def __call__(self):
        now = self.clock()
        self._attempts = [t for t in self._attempts if (now - t).total_seconds() < self.window]
        if len(self._attempts) >= self.limit_rpm:
            self.n429 += 1
            return False
        self._attempts.append(now)
        return True


def _limiter(clock, **kw):
    async def fake_sleep(secs):
        clock.advance(secs)

    kw.setdefault("default_rpm", 25.0)
    kw.setdefault("persist_path", "")
    return AdaptiveRateLimiter("test", now_provider=clock, async_sleep=fake_sleep, **kw)


async def _drive(limiter, provider, clock, steps, think=1.0):
    """Send ``steps`` requests through the limiter; return list of 200/429."""
    out = []
    for _ in range(steps):
        await limiter.acquire("m")
        ok = provider()
        if ok:
            limiter.on_success("m")
        else:
            limiter.on_429("m")
        clock.advance(think)
        out.append(ok)
    return out


# --- direct controller math -------------------------------------------------
def test_429_lowers_estimate():
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, default_rpm=25.0)
    st = limiter._state_for("m")
    for _ in range(6):
        st.window.append(clock())
    backoff = limiter.on_429("m")
    assert st.rpm < 25.0          # backed off
    assert backoff == 5.0         # base cooldown


def test_success_raises_estimate_when_stable():
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, default_rpm=5.0)
    st = limiter._state_for("m")
    st.last_429 = clock() - timedelta(minutes=10)  # stable
    before = st.rpm
    limiter.on_success("m")
    assert st.rpm == before + 1.0


def test_cooldown_grows_with_repeated_429():
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, default_rpm=25.0)
    st = limiter._state_for("m")
    st.window.append(clock())
    assert limiter.on_429("m") == 5.0
    assert limiter.on_429("m") == 10.0
    assert limiter.on_429("m") == 20.0


def test_rechallenge_probes_higher():
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, default_rpm=5.0)
    st = limiter._state_for("m")
    st.rpm = 3.0
    st.last_rechallenge = clock() - timedelta(hours=25)
    before = st.rpm
    assert limiter._maybe_rechallenge(st, clock()) is True
    assert st.rpm > before
    assert st.ceiling == limiter.ceiling_rpm


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "rl.json")
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, persist_path=path, default_rpm=5.0)
    limiter._state_for("m").rpm = 7.3
    limiter._save()
    reloaded = _limiter(FakeClock(datetime(2026, 1, 1)), persist_path=path, default_rpm=5.0)
    assert reloaded._state_for("m").rpm == pytest.approx(7.3)


def test_parse_retry_after():
    from uniinfer.providers.tu import _parse_retry_after

    assert _parse_retry_after({"retry-after": "30"}) == 30.0
    assert _parse_retry_after({}) is None


def test_staging_provider_uses_separate_limiter():
    """TUStagingProvider must not share production TU's learned rate limit."""
    from uniinfer.providers.tu import TUProvider, TUStagingProvider

    prod = TUProvider(api_key="k")
    staging = TUStagingProvider(api_key="k")
    assert prod._rate_limiter() is not staging._rate_limiter()
    assert prod._rate_limiter().provider_id == "tu"
    assert staging._rate_limiter().provider_id == "tu-staging"


def test_from_dict_tolerates_bad_values(tmp_path):
    """A corrupt/partial state file must not crash the load path."""
    import json

    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        json.dump({"providers": {"test": {"m": {"rpm": "not-a-number", "ceiling": None}}}}, f)
    limiter = _limiter(FakeClock(datetime(2026, 1, 1)), persist_path=path, default_rpm=5.0)
    assert limiter._state_for("m").rpm == 5.0  # fell back to default, no crash


def test_all_rate_limiter_status_aggregates_providers():
    """all_rate_limiter_status() exposes every provider's per-model state."""
    a = get_rate_limiter("agg-a", default_rpm=10.0)
    b = get_rate_limiter("agg-b", default_rpm=20.0)
    a._state_for("m1")
    b._state_for("m2")
    status = all_rate_limiter_status()
    assert "agg-a" in status and "agg-b" in status
    assert "m1" in status["agg-a"]
    assert "m2" in status["agg-b"]


# --- end-to-end simulation --------------------------------------------------
@pytest.mark.asyncio
async def test_adaptation_beats_fixed_high_rate():
    """A 5/min backend: adaptive limiter sees far fewer 429s than a pinned-high one."""
    # Adaptive.
    clock = FakeClock(datetime(2026, 1, 1))
    adaptive = await _drive(_limiter(clock, default_rpm=25.0), FakeProvider(clock, 5), clock, 80)

    # Fixed high rate (adaptation disabled: on_429 is a no-op, rpm pinned high).
    clock = FakeClock(datetime(2026, 1, 1))
    limiter = _limiter(clock, default_rpm=25.0)
    limiter.on_429 = lambda *a, **k: 0.0  # type: ignore[assignment]
    fixed = await _drive(limiter, FakeProvider(clock, 5), clock, 80)

    assert adaptive.count(False) < fixed.count(False) / 2
    assert adaptive[-20:].count(False) <= 4   # clean tail once learned


@pytest.mark.asyncio
async def test_discovers_limit_upgrade():
    """Backend raised 5 -> 40: the limiter climbs well above its old ceiling."""
    clock = FakeClock(datetime(2026, 1, 1))
    provider = FakeProvider(clock, 5)
    limiter = _limiter(clock, default_rpm=25.0)
    limiter.stable_threshold = timedelta(seconds=1)

    await _drive(limiter, provider, clock, 40)          # learn 5/min
    assert limiter._state_for("m").rpm <= 8.0

    provider.set_limit(40)                              # silent upgrade
    await _drive(limiter, provider, clock, 400)
    assert limiter._state_for("m").rpm > 20.0           # re-discovered headroom
