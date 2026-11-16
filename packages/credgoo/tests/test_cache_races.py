"""Regression tests for the two get()/add() cache races (issue
.work/issues/backlog/credgoo-cache-races.md, found via TLA+ 2026-11-16).

Both races share one root cause: add() runs backend-write -> L1-invalidate ->
L2-re-prime as three unguarded steps, while get() probes L1/L2 on hot-loop
threads. The fixes under test:

* a per-service lock serializes get() against the whole add()/delete()
  sequence (closes the mid-write window, Race 1);
* a process-global generation counter makes L1 re-poisoning structurally
  impossible: mutations bump it, get() captures it on entry and stamps it into
  every L1 entry it writes, and reads discard entries from an older
  generation (kills Race 2 even if a path ever bypasses the lock).

Determinism: interleaving is forced with threading.Event gates inside a fake
backend / a wrapped cache write — no sleeps on the code under test. Every
racy get() runs in its own thread, and the test asserts the thread CANNOT
finish while the mutation holds the window open (a pre-fix build lets it
return the stale key immediately, which trips the same assert).
Hermetic: no network, no tokens, no real backend module (the fake registers
into credgoo.store.BACKENDS for the duration of a test).
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import credgoo.store as store
from credgoo.store import (
    CredentialStore,
    _cache_key_name,
    _l1_clear,
)

WINDOW_PROBE_SECONDS = 0.5


class FakeBackend:
    """In-memory backend with event gates to pin add() mid-flight.

    Class-level state on purpose: the CLI constructs a CredentialStore per
    operation, so probe and writer go through DIFFERENT store instances that
    must still observe the same backend (that is the production shape).
    """

    name = "fake"
    cache_integrity = False
    keys = {}
    backend_committed = threading.Event()
    release_add = threading.Event()

    def setup(self, cache_dir):
        return True

    def cache_key(self, creds):
        return creds.get("encryption_key")

    def fetch_key(self, service, creds):
        return self.keys.get(service)

    def supports(self, capability):
        return True

    def add_key(self, service, api_key, creds):
        self.keys[service] = api_key
        self.backend_committed.set()
        self.release_add.wait(timeout=10)
        return True

    def delete_key(self, service, creds):
        return self.keys.pop(service, None) is not None


def _spawn_get(store_obj, service):
    """Run store.get(service) on its own thread; report result + completion."""
    result = {}
    done = threading.Event()

    def run():
        result["value"] = store_obj.get(service)
        done.set()

    thread = threading.Thread(target=run)
    return thread, result, done


class CacheRaceTests(unittest.TestCase):
    def setUp(self):
        _l1_clear()
        FakeBackend.keys = {}
        FakeBackend.backend_committed = threading.Event()
        FakeBackend.release_add = threading.Event()
        backend_patch = patch.dict(store.BACKENDS, {"fake": FakeBackend})
        backend_patch.start()
        self.addCleanup(backend_patch.stop)
        self.addCleanup(_l1_clear)

    def _store(self, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "credgoo.txt").write_text(json.dumps(
            {"default_backend": "fake", "fake": {"encryption_key": "enc123"}}))
        return CredentialStore(cache_dir)

    def test_race1_get_during_add_never_serves_old_key(self):
        """Mid-write window: backend holds NEW, add() not yet returned — a
        concurrent get() must not serve OLD. Pre-fix this reproduced
        instantly: get() hit L1/L2 with the OLD key mid-sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            FakeBackend.keys["svc"] = "OLD"
            probe = self._store(tmp)
            self.assertEqual(probe.get("svc"), "OLD")

            writer = self._store(tmp)
            t_add = threading.Thread(target=lambda: writer.add("svc", "NEW"))
            t_add.start()
            self.assertTrue(FakeBackend.backend_committed.wait(timeout=10))

            t_get, result, get_done = _spawn_get(probe, "svc")
            t_get.start()
            # Pre-fix the get returned OLD immediately; a fixed build keeps it
            # blocked on the per-service lock until add() finishes.
            self.assertFalse(
                get_done.wait(timeout=WINDOW_PROBE_SECONDS),
                "get() completed mid-add — window is open again")
            FakeBackend.release_add.set()
            t_add.join(timeout=10)
            t_get.join(timeout=10)

            self.assertEqual(result["value"], "NEW")
            self.assertEqual(probe.get("svc"), "NEW")

    def test_race2_l1_reprime_does_not_outlive_completed_add(self):
        """L1 re-poisoning: a get() between L1-invalidation and L2-re-prime
        must not leave the dead key in L1 past add()'s return. Pre-fix the
        racy read re-primed L1 with OLD2 and every later get() served it for
        the whole 60s L1 TTL."""
        with tempfile.TemporaryDirectory() as tmp:
            FakeBackend.keys["svc"] = "OLD2"
            probe = self._store(tmp)
            self.assertEqual(probe.get("svc"), "OLD2")
            FakeBackend.keys["svc"] = "NEW2"

            real_cache = store.cache_api_key
            entered = threading.Event()
            release = threading.Event()

            def gated_cache(*args, **kwargs):
                entered.set()  # add() is now between _l1_invalidate and the L2 write
                release.wait(timeout=10)
                return real_cache(*args, **kwargs)

            writer = self._store(tmp)
            FakeBackend.release_add.set()  # the gate here is cache_api_key, not add_key
            with patch.object(store, "cache_api_key", gated_cache):
                t_add = threading.Thread(target=lambda: writer.add("svc", "NEW2"))
                t_add.start()
                self.assertTrue(entered.wait(timeout=10))

                t_get, result, get_done = _spawn_get(probe, "svc")
                t_get.start()
                self.assertFalse(
                    get_done.wait(timeout=WINDOW_PROBE_SECONDS),
                    "get() completed inside the invalidation window — "
                    "re-poisoning is possible again")
                release.set()
                t_add.join(timeout=10)
                t_get.join(timeout=10)

            self.assertEqual(result["value"], "NEW2")
            after = probe.get("svc")  # add() long finished, L2 holds NEW2
            self.assertEqual(after, "NEW2")

    def test_generation_stamp_discards_pre_mutation_l1_entry(self):
        """The generation counter alone (no lock) must kill re-poisoning: an
        L1 entry stamped before a mutation is discarded on the next read even
        if it never expired and was never explicitly invalidated."""
        with tempfile.TemporaryDirectory() as tmp:
            FakeBackend.keys["svc"] = "OLD"
            probe = self._store(tmp)
            self.assertEqual(probe.get("svc"), "OLD")

            # Simulate the racy re-prime: an entry stamped with the PRE-mutation
            # generation planted after the bump (what the lock normally prevents).
            name = probe._l1_name(_cache_key_name("fake", "svc"))
            from credgoo.store import _l1_current_generation
            pre_mutation_gen = _l1_current_generation()
            FakeBackend.keys["svc"] = "NEW"
            writer = self._store(tmp)
            with patch.object(FakeBackend, "add_key", return_value=True):
                self.assertTrue(writer.add("svc", "NEW"))
            store._L1_CACHE[name] = ("OLD", time.time(), pre_mutation_gen)

            self.assertEqual(probe.get("svc"), "NEW")

    def test_delete_bumps_generation_and_l1_never_serves_dead_key(self):
        """delete() must close the same windows as add(): a get() racing the
        delete sequence may not keep serving the deleted key from L1."""
        with tempfile.TemporaryDirectory() as tmp:
            FakeBackend.keys["svc"] = "DOOMED"
            probe = self._store(tmp)
            self.assertEqual(probe.get("svc"), "DOOMED")

            writer = self._store(tmp)
            entered = threading.Event()
            release = threading.Event()
            real_delete = FakeBackend.delete_key

            def gated_delete(self_be, service, creds):
                ok = real_delete(self_be, service, creds)
                entered.set()  # backend deleted, caches not yet updated
                release.wait(timeout=10)
                return ok

            with patch.object(FakeBackend, "delete_key", gated_delete):
                t_del = threading.Thread(target=lambda: writer.delete("svc"))
                t_del.start()
                self.assertTrue(entered.wait(timeout=10))
                FakeBackend.keys["svc"] = None  # a backend fetch would fail now

                t_get, result, get_done = _spawn_get(probe, "svc")
                t_get.start()
                self.assertFalse(
                    get_done.wait(timeout=WINDOW_PROBE_SECONDS),
                    "get() completed mid-delete — window is open again")
                release.set()
                t_del.join(timeout=10)
                t_get.join(timeout=10)

            self.assertIsNone(result["value"])  # not DOOMED, and no zombie fetch


if __name__ == "__main__":
    unittest.main()
