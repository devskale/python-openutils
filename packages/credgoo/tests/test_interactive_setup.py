import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from credgoo import credgoo as mod
from credgoo.backends.gdrive import GdriveBackend
from credgoo.backends.airtable import AirtableBackend


class TestCredentialMigration(unittest.TestCase):
    def test_v1_flat_gdrive_migrates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"url": "https://script.google.com/xxx", "token": "t", "encryption_key": "k"}, f)
            creds = mod.load_credentials(cred_file)
            self.assertEqual(creds["default_backend"], "gdrive")
            self.assertEqual(creds["gdrive"]["url"], "https://script.google.com/xxx")
            self.assertEqual(creds["gdrive"]["token"], "t")
            self.assertEqual(creds["gdrive"]["encryption_key"], "k")

    def test_v2_single_backend_migrates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            with open(cred_file, "w") as f:
                json.dump({"backend": "gdrive", "url": "u", "token": "t", "encryption_key": "k"}, f)
            creds = mod.load_credentials(cred_file)
            self.assertEqual(creds["default_backend"], "gdrive")
            self.assertEqual(creds["gdrive"]["url"], "u")

    def test_v3_multi_backend_passes_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            data = {"default_backend": "gdrive", "gdrive": {"url": "u", "token": "t", "encryption_key": "k"}}
            with open(cred_file, "w") as f:
                json.dump(data, f)
            creds = mod.load_credentials(cred_file)
            self.assertEqual(creds, data)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            creds = mod.load_credentials(cred_file)
            self.assertEqual(creds, {})


class TestBackendResolution(unittest.TestCase):
    def test_resolves_default(self):
        creds = {"default_backend": "gdrive", "gdrive": {"url": "u"}}
        name, backend, backend_creds = mod._resolve_backend(creds)
        self.assertEqual(name, "gdrive")
        self.assertIsInstance(backend, GdriveBackend)
        self.assertEqual(backend_creds, {"url": "u"})

    def test_resolves_explicit_override(self):
        creds = {"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t"}}
        name, backend, backend_creds = mod._resolve_backend(creds, "airtable")
        self.assertEqual(name, "airtable")
        self.assertIsInstance(backend, AirtableBackend)
        self.assertEqual(backend_creds, {"airtable_token": "t"})

    def test_no_default_returns_none(self):
        name, backend, backend_creds = mod._resolve_backend({})
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
            creds = mod.load_credentials(cred_file)
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
            mod.store_credentials(
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
            mod.store_credentials(
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
            mod.store_credentials(
                {"default_backend": "gdrive", "gdrive": {"url": "u"}, "airtable": {"airtable_token": "t", "airtable_base": "b"}},
                cache_dir / "credgoo.txt",
            )
            with patch.object(AirtableBackend, "fetch_key", return_value="sk-at") as mock:
                result = mod.get_api_key("openai", cache_dir=cache_dir, no_cache=True, backend_name="airtable")
            self.assertEqual(result, "sk-at")
            mock.assert_called_once()


class TestCacheNamespacing(unittest.TestCase):
    def test_cache_key_namespaced(self):
        self.assertEqual(mod._cache_key_name("gdrive", "openai"), "gdrive:openai")
        self.assertEqual(mod._cache_key_name("airtable", "openai"), "airtable:openai")


class TestStoreLoadCredentials(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = Path(tmpdir) / "credgoo.txt"
            data = {"default_backend": "airtable", "airtable": {"airtable_token": "secret"}}
            mod.store_credentials(data, cred_file)
            loaded = mod.load_credentials(cred_file)
            self.assertEqual(loaded, data)


class TestDerivedKeyFromPAT(unittest.TestCase):
    def test_derive_key_deterministic(self):
        key1 = mod._derive_key_from_pat("patABC123")
        key2 = mod._derive_key_from_pat("patABC123")
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 32)

    def test_different_pats_different_keys(self):
        key1 = mod._derive_key_from_pat("patABC123")
        key2 = mod._derive_key_from_pat("patXYZ789")
        self.assertNotEqual(key1, key2)

    def test_get_enc_key_airtable_derives_from_pat(self):
        key = mod._get_enc_key("airtable", {"airtable_token": "patABC123"})
        self.assertEqual(key, mod._derive_key_from_pat("patABC123"))

    def test_get_enc_key_gdrive_uses_stored(self):
        key = mod._get_enc_key("gdrive", {"encryption_key": "mykey123"})
        self.assertEqual(key, "mykey123")

    def test_get_enc_key_airtable_no_pat(self):
        key = mod._get_enc_key("airtable", {})
        self.assertIsNone(key)


class TestHMACValidation(unittest.TestCase):
    def test_hmac_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            enc_key = mod._derive_key_from_pat("patTEST")
            mod.cache_api_key("airtable", "openai", "sk-123", enc_key, cache_dir)
            result = mod.get_cached_api_key("airtable", "openai", enc_key, cache_dir)
            self.assertEqual(result, "sk-123")

    def test_hmac_mismatch_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_key = mod._derive_key_from_pat("patOLD")
            new_key = mod._derive_key_from_pat("patNEW")

            # Cache with old key
            mod.cache_api_key("airtable", "openai", "sk-123", old_key, cache_dir)

            # Try to read with new key (simulating PAT rotation)
            result = mod.get_cached_api_key("airtable", "openai", new_key, cache_dir)
            self.assertIsNone(result)

            # Verify the stale entry was removed from cache
            cache_file = cache_dir / "api_keys.json"
            with open(cache_file) as f:
                cache = json.load(f)
            self.assertNotIn("airtable:openai", cache)

    def test_gdrive_no_hmac_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            mod.cache_api_key("gdrive", "openai", "sk-123", "mykey", cache_dir)
            cache_file = cache_dir / "api_keys.json"
            with open(cache_file) as f:
                entry = json.load(f)["gdrive:openai"]
            self.assertNotIn("hmac", entry)

    def test_airtable_hmac_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            enc_key = mod._derive_key_from_pat("patTEST")
            mod.cache_api_key("airtable", "openai", "sk-123", enc_key, cache_dir)
            cache_file = cache_dir / "api_keys.json"
            with open(cache_file) as f:
                entry = json.load(f)["airtable:openai"]
            self.assertIn("hmac", entry)


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


if __name__ == "__main__":
    unittest.main()
