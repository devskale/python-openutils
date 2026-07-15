import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from credgoo import credgoo as mod
from credgoo.store import (
    CredentialStore,
    _cache_key_name,
    _resolve_backend,
    cache_api_key,
    get_cached_api_key,
    load_credentials,
    store_credentials,
)
from credgoo.backends.base import CredgooBackend, UnsupportedOperation
from credgoo.backends.gdrive import GdriveBackend
from credgoo.backends.airtable import AirtableBackend, AirtableClient


class TestCredentialMigration(unittest.TestCase):
    def test_v1_flat_gdrive_migrates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"url": "https://script.google.com/xxx", "token": "t", "encryption_key": "k"}, f)
            creds = load_credentials(cred_file)
            self.assertEqual(creds["default_backend"], "gdrive")
            self.assertEqual(creds["gdrive"]["url"], "https://script.google.com/xxx")
            self.assertEqual(creds["gdrive"]["token"], "t")
            self.assertEqual(creds["gdrive"]["encryption_key"], "k")

    def test_v2_single_backend_migrates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"backend": "gdrive", "url": "u", "token": "t", "encryption_key": "k"}, f)
            creds = load_credentials(cred_file)
            self.assertEqual(creds["default_backend"], "gdrive")
            self.assertEqual(creds["gdrive"]["url"], "u")

    def test_v3_multi_backend_passes_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            data = {"default_backend": "gdrive", "gdrive": {"url": "u", "token": "t", "encryption_key": "k"}}
            with open(cred_file, "w") as f:
                json.dump(data, f)
            creds = load_credentials(cred_file)
            self.assertEqual(creds, data)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            creds = load_credentials(cred_file)
            self.assertEqual(creds, {})


class TestBackendResolution(unittest.TestCase):
    def test_resolves_default(self):
        creds = {"default_backend": "gdrive", "gdrive": {"url": "u"}}
        name, backend, backend_creds = _resolve_backend(creds)
        self.assertEqual(name, "gdrive")
        self.assertIsInstance(backend, GdriveBackend)
        self.assertEqual(backend_creds, {"url": "u"})

    def test_resolves_explicit_override(self):
        creds = {"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t"}}
        name, backend, backend_creds = _resolve_backend(creds, "airtable")
        self.assertEqual(name, "airtable")
        self.assertIsInstance(backend, AirtableBackend)
        self.assertEqual(backend_creds, {"airtable_token": "t"})

    def test_no_default_returns_none(self):
        name, backend, backend_creds = _resolve_backend({})
        self.assertIsNone(name)
        self.assertIsNone(backend)


class TestSetDefaultBackend(unittest.TestCase):
    def test_set_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cred_file = cache_dir / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t"}}, f)
            result = mod.set_default_backend("airtable", cache_dir)
            self.assertTrue(result)
            creds = load_credentials(cred_file)
            self.assertEqual(creds["default_backend"], "airtable")

    def test_set_unknown_backend_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mod.set_default_backend("nonexistent", Path(tmpdir))
            self.assertFalse(result)

    def test_set_unconfigured_backend_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cred_file = cache_dir / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"default_backend": "gdrive", "gdrive": {"url": "u"}}, f)
            result = mod.set_default_backend("airtable", cache_dir)
            self.assertFalse(result)


class TestAirtableClient(unittest.TestCase):
    """The client absorbs pagination and the cred guard — testable in isolation,
    which the backend's record methods were not before."""

    def _resp(self, body, status=200):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = body
        return r

    def test_unconfigured_when_creds_missing(self):
        self.assertFalse(AirtableClient({}).configured)

    def test_configured_with_token_and_base(self):
        self.assertTrue(AirtableClient({"airtable_token": "t", "airtable_base": "b"}).configured)

    def test_iter_records_follows_offset_pagination(self):
        client = AirtableClient({"airtable_token": "t", "airtable_base": "b"})
        page1 = self._resp({"records": [
            {"id": "r1", "fields": {"service": "a"}},
            {"id": "r2", "fields": {"service": "b"}}], "offset": "NEXT"})
        page2 = self._resp({"records": [{"id": "r3", "fields": {"service": "c"}}]})
        with patch("credgoo.backends.airtable.requests.get", side_effect=[page1, page2]) as mock:
            ids = [r["id"] for r in client.iter_records(["service"])]
        self.assertEqual(ids, ["r1", "r2", "r3"])
        self.assertEqual(mock.call_count, 2)

    def test_list_keys_paginates_through_backend(self):
        backend = AirtableBackend()
        page1 = self._resp({"records": [{"id": "r1", "fields": {"service": "openai"}}], "offset": "NEXT"})
        page2 = self._resp({"records": [{"id": "r2", "fields": {"service": "groq"}}]})
        with patch("credgoo.backends.airtable.requests.get", side_effect=[page1, page2]):
            self.assertEqual(backend.list_keys({"airtable_token": "t", "airtable_base": "b"}), ["groq", "openai"])

    def test_dedupe_batches_deletes_through_backend(self):
        backend = AirtableBackend()
        page = self._resp({"records": [
            {"id": "r1", "fields": {"service": "openai"}},
            {"id": "r2", "fields": {"service": "openai"}},
            {"id": "r3", "fields": {"service": "groq"}}]})
        with patch("credgoo.backends.airtable.requests.get", return_value=page), \
                patch("credgoo.backends.airtable.requests.delete", return_value=self._resp({})) as mock_del:
            kept, removed = backend.dedupe_keys({"airtable_token": "t", "airtable_base": "b"})
        self.assertEqual((kept, removed), (2, 1))
        mock_del.assert_called_once()  # 2 dups fit one batched call


class TestGdriveBackend(unittest.TestCase):
    def test_fetch_key_decrypts(self):
        backend = GdriveBackend()
        creds = {"url": "https://script.google.com/xxx", "token": "test-token", "encryption_key": "testkey123"}
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"status": "success", "encryptedKey": "AAAAAAAAAAAAIPDIyQ=="}
        with patch("credgoo.backends.gdrive.requests.get", return_value=fake_response):
            key = backend.fetch_key("openai", creds)
        self.assertIsNotNone(key)

    def test_fetch_key_missing_creds(self):
        backend = GdriveBackend()
        self.assertIsNone(backend.fetch_key("openai", {}))

    def test_fetch_key_api_error(self):
        backend = GdriveBackend()
        creds = {"url": "u", "token": "t", "encryption_key": "k"}
        fake_response = MagicMock()
        fake_response.status_code = 500
        fake_response.text = "error"
        with patch("credgoo.backends.gdrive.requests.get", return_value=fake_response):
            self.assertIsNone(backend.fetch_key("openai", creds))


class TestAirtableBackend(unittest.TestCase):
    def test_fetch_key_returns_value(self):
        backend = AirtableBackend()
        creds = {"airtable_token": "patxxx", "airtable_base": "appXXX", "airtable_table": "keys"}
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"records": [{"fields": {"key": "sk-abc123"}}]}
        with patch("credgoo.backends.airtable.requests.get", return_value=fake_response):
            key = backend.fetch_key("openai", creds)
        self.assertEqual(key, "sk-abc123")

    def test_fetch_key_missing_creds(self):
        backend = AirtableBackend()
        self.assertIsNone(backend.fetch_key("openai", {}))

    def test_fetch_key_not_found(self):
        backend = AirtableBackend()
        creds = {"airtable_token": "t", "airtable_base": "b"}
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"records": []}
        with patch("credgoo.backends.airtable.requests.get", return_value=fake_response):
            self.assertIsNone(backend.fetch_key("nonexistent", creds))


class TestGetApiKeyDispatches(unittest.TestCase):
    def test_dispatches_to_gdrive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "gdrive", "gdrive": {"url": "u", "token": "t", "encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            with patch.object(GdriveBackend, "fetch_key", return_value="sk-test") as mock:
                result = mod.get_api_key("openai", cache_dir=cache_dir, no_cache=True)
            self.assertEqual(result, "sk-test")
            mock.assert_called_once()

    def test_dispatches_to_airtable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "airtable", "airtable": {"airtable_token": "t", "airtable_base": "b", "encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            with patch.object(AirtableBackend, "fetch_key", return_value="sk-at") as mock:
                result = mod.get_api_key("openai", cache_dir=cache_dir, no_cache=True)
            self.assertEqual(result, "sk-at")
            mock.assert_called_once()

    def test_backend_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t", "airtable_base": "b"}},
                cache_dir / "credgoo.txt",
            )
            with patch.object(AirtableBackend, "fetch_key", return_value="sk-at") as mock:
                result = mod.get_api_key("openai", cache_dir=cache_dir, no_cache=True, backend_name="airtable")
            self.assertEqual(result, "sk-at")
            mock.assert_called_once()


class TestCacheNamespacing(unittest.TestCase):
    def test_cache_key_namespaced(self):
        self.assertEqual(_cache_key_name("gdrive", "openai"), "gdrive:openai")
        self.assertEqual(_cache_key_name("airtable", "openai"), "airtable:openai")


class TestStoreLoadCredentials(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            data = {"default_backend": "airtable", "airtable": {"airtable_token": "secret"}}
            store_credentials(data, cred_file)
            loaded = load_credentials(cred_file)
            self.assertEqual(loaded, data)


class TestCacheKeyDerivation(unittest.TestCase):
    """The cache-encryption key is derived by the backend, not the store."""

    def test_airtable_derives_deterministically(self):
        k1 = AirtableBackend().cache_key({"airtable_token": "patABC123"})
        k2 = AirtableBackend().cache_key({"airtable_token": "patABC123"})
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 32)

    def test_airtable_different_tokens_different_keys(self):
        self.assertNotEqual(
            AirtableBackend().cache_key({"airtable_token": "patABC123"}),
            AirtableBackend().cache_key({"airtable_token": "patXYZ789"}))

    def test_gdrive_uses_stored_key(self):
        self.assertEqual(GdriveBackend().cache_key({"encryption_key": "mykey123"}), "mykey123")

    def test_airtable_no_token_returns_none(self):
        self.assertIsNone(AirtableBackend().cache_key({}))


class TestCacheIntegrityMechanism(unittest.TestCase):
    """The cache is backend-agnostic: an opt-in *integrity* flag drives the HMAC
    tag, not a backend name. These exercise the mechanism directly."""

    def test_integrity_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            enc_key = AirtableBackend().cache_key({"airtable_token": "patTEST"})
            cache_api_key("airtable", "openai", "sk-123", enc_key, cache_dir, integrity=True)
            result = get_cached_api_key("airtable", "openai", enc_key, cache_dir, integrity=True)
            self.assertEqual(result, "sk-123")

    def test_integrity_mismatch_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_key = AirtableBackend().cache_key({"airtable_token": "patOLD"})
            new_key = AirtableBackend().cache_key({"airtable_token": "patNEW"})

            cache_api_key("airtable", "openai", "sk-123", old_key, cache_dir, integrity=True)
            result = get_cached_api_key("airtable", "openai", new_key, cache_dir, integrity=True)
            self.assertIsNone(result)

            cache_file = cache_dir / "api_keys.json"
            with open(cache_file) as f:
                cache = json.load(f)
            self.assertNotIn("airtable:openai", cache)

    def test_integrity_off_stores_no_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_api_key("gdrive", "openai", "sk-123", "mykey", cache_dir)  # default integrity=False
            with open(cache_dir / "api_keys.json") as f:
                entry = json.load(f)["gdrive:openai"]
            self.assertNotIn("hmac", entry)

    def test_integrity_decoupled_from_backend_name(self):
        # Regression for #4: an arbitrary name (not "airtable") gets the HMAC tag
        # purely from the flag. Before #4 this stored no tag (name-based dispatch).
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_api_key("anyname", "openai", "sk-123", "somekey", cache_dir, integrity=True)
            result = get_cached_api_key("anyname", "openai", "somekey", cache_dir, integrity=True)
            self.assertEqual(result, "sk-123")
            with open(cache_dir / "api_keys.json") as f:
                self.assertIn("hmac", json.load(f)["anyname:openai"])


class TestCacheIntegrityPolicy(unittest.TestCase):
    """The integrity *policy* lives on the backend; the cache just honours it."""

    def test_gdrive_has_no_integrity_by_default(self):
        self.assertFalse(GdriveBackend().cache_integrity)

    def test_airtable_opts_into_integrity(self):
        self.assertTrue(AirtableBackend().cache_integrity)

    def test_policy_flows_through_store_to_refresh_on_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cred_file = cache_dir / "credgoo.txt"
            # airtable under PAT A — caches a key, HMAC-stamped under PAT A
            store_credentials(
                {"default_backend": "airtable",
                 "airtable": {"airtable_token": "patOLD", "airtable_base": "b"}},
                cred_file)
            with patch.object(AirtableBackend, "fetch_key", return_value="sk-1"):
                self.assertEqual(CredentialStore(cache_dir).get("openai"), "sk-1")
            # rotate the PAT — stale cache must not be served
            store_credentials(
                {"default_backend": "airtable",
                 "airtable": {"airtable_token": "patNEW", "airtable_base": "b"}},
                cred_file)
            with patch.object(AirtableBackend, "fetch_key", return_value="sk-rotated") as mock:
                self.assertEqual(CredentialStore(cache_dir).get("openai"), "sk-rotated")
            mock.assert_called_once()


class TestCredentialStore(unittest.TestCase):
    """Exercises the deep module through its own (data-only) interface.

    No requests, no stdout — a fake backend stands in for the real seam, so the
    cache read/write/invalidate orchestration is tested directly.
    """

    def test_unconfigured_store_reports_not_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CredentialStore(Path(tmpdir))
            self.assertFalse(store.configured)
            self.assertIsNone(store.get("anything"))

    def test_gdrive_supports_only_fetch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "gdrive", "gdrive": {"url": "u", "token": "t", "encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            store = CredentialStore(cache_dir)
            self.assertTrue(store.configured)
            self.assertTrue(store.supports("fetch_key"))
            self.assertFalse(store.supports("add_key"))
            self.assertFalse(store.supports("delete_key"))

    def test_get_serves_second_call_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "fake", "fake": {"encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            with patch.dict("credgoo.store.BACKENDS", {"fake": _FakeBackend}):
                with patch.object(_FakeBackend, "fetch_key", return_value="sk-1") as mock_fetch:
                    store = CredentialStore(cache_dir)
                    self.assertEqual(store.get("openai"), "sk-1")
                    self.assertEqual(store.get("openai"), "sk-1")  # served from cache
            self.assertEqual(mock_fetch.call_count, 1)  # backend hit once, cache hit once

    def test_add_primes_cache_and_delete_invalidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "fake", "fake": {"encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            with patch.dict("credgoo.store.BACKENDS", {"fake": _FakeBackend}):
                store = CredentialStore(cache_dir)
                self.assertTrue(store.add("openai", "sk-1"))       # primes cache
                self.assertEqual(store.get("openai"), "sk-1")      # present
                self.assertTrue(store.delete("openai"))            # invalidate

                # New store, same cache dir: backend has no key, cache must be gone
                store2 = CredentialStore(cache_dir)
                self.assertIsNone(store2.get("openai"))

    def test_list_and_dedupe_through_interface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "fake", "fake": {"encryption_key": "k"}},
                cache_dir / "credgoo.txt",
            )
            with patch.dict("credgoo.store.BACKENDS", {"fake": _FakeBackend}):
                store = CredentialStore(cache_dir)
                store.add("openai", "sk-1")
                store.add("groq", "gsk-2")
                self.assertEqual(store.list(), ["groq", "openai"])
                self.assertEqual(store.dedupe(), (2, 0))

    def test_set_default_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            store_credentials(
                {"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t"}},
                cache_dir / "credgoo.txt",
            )
            store = CredentialStore(cache_dir)
            self.assertTrue(store.set_default("airtable"))
            self.assertEqual(load_credentials(cache_dir / "credgoo.txt")["default_backend"], "airtable")
            self.assertFalse(store.set_default("nonexistent"))


class TestBackendContract(unittest.TestCase):
    """The ABC now carries the full contract; capability is an override, not a sniff."""

    def test_gdrive_supports_fetch_only(self):
        b = GdriveBackend()
        self.assertTrue(b.supports("fetch_key"))
        for cap in ("add_key", "delete_key", "clear_key", "list_keys", "dedupe_keys"):
            self.assertFalse(b.supports(cap), f"gdrive should not support {cap}")

    def test_airtable_supports_every_capability(self):
        b = AirtableBackend()
        for cap in ("fetch_key", "add_key", "delete_key", "clear_key", "list_keys", "dedupe_keys"):
            self.assertTrue(b.supports(cap), f"airtable should support {cap}")

    def test_unsupported_call_raises_typed(self):
        b = GdriveBackend()
        with self.assertRaises(UnsupportedOperation) as ctx:
            b.add_key("openai", "sk", {})
        self.assertEqual(ctx.exception.capability, "add_key")
        self.assertEqual(ctx.exception.backend_name, "gdrive")

    def test_every_capability_names_a_real_method(self):
        # the contract guarantees these exist on the type — no hasattr probing needed
        for cap in ("setup", "fetch_key", "add_key", "delete_key", "clear_key", "list_keys", "dedupe_keys"):
            self.assertTrue(hasattr(CredgooBackend, cap))


class TestMainCli(unittest.TestCase):
    def test_setup_backend_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "--setup-backend", "--cache-dir", tmpdir]), \
                    patch.object(mod.sys.stdin, "isatty", return_value=True), \
                    patch.object(mod, "setup_backend", return_value=True):
                self.assertEqual(mod.main(), 0)

    def test_set_backend_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cred_file = cache_dir / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t"}}, f)
            with patch.object(mod.sys, "argv", ["credgoo", "--set-backend", "airtable", "--cache-dir", tmpdir]):
                self.assertEqual(mod.main(), 0)

    def test_retrieves_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "openai", "--cache-dir", tmpdir]), \
                    patch.object(mod, "get_api_key", return_value="sk-123"):
                self.assertEqual(mod.main(), 0)

    def test_backend_flag_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "openai", "--backend", "airtable", "--cache-dir", tmpdir]), \
                    patch.object(mod, "get_api_key", return_value="sk-at") as mock:
                mod.main()
            mock.assert_called_once_with("openai", cache_dir=Path(tmpdir), no_cache=False, backend_name="airtable")


class _FakeBackend(CredgooBackend):
    """In-memory backend for store tests — a real CredgooBackend that implements
    the full contract by overriding, exactly as a third backend would."""

    name = "fake"

    def __init__(self):
        self.keys = {}

    def setup(self, cache_dir):
        return True

    def fetch_key(self, service, creds):
        return self.keys.get(service)

    def add_key(self, service, api_key, creds):
        self.keys[service] = api_key
        return True

    def delete_key(self, service, creds):
        self.keys.pop(service, None)
        return True

    def clear_key(self, service, creds):
        self.keys.pop(service, None)
        return True

    def list_keys(self, creds):
        return sorted(self.keys)

    def dedupe_keys(self, creds):
        return len(self.keys), 0


if __name__ == "__main__":
    unittest.main()
