import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from credgoo import credgoo as mod


class TestInteractiveSetup(unittest.TestCase):
    def test_interactive_setup_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            stored = []
            cached = []

            def fake_store(token, encryption_key, url, cred_file, save_token_flag, save_key_flag, save_url_flag):
                stored.append((token, encryption_key, url, cred_file, save_token_flag, save_key_flag, save_url_flag))

            def fake_cache(service, api_key, encryption_key, cache_dir_param):
                cached.append((service, api_key, encryption_key, cache_dir_param))

            with patch("builtins.input", side_effect=["https://example.com/script"]), \
                    patch.object(mod.getpass, "getpass", side_effect=["secret-value", "secret-value"]), \
                    patch.object(mod, "get_api_key_from_google", return_value="api-key-123"), \
                    patch.object(mod, "store_credentials", side_effect=fake_store), \
                    patch.object(mod, "cache_api_key", side_effect=fake_cache):
                token, key, url, api_key, ok = mod.interactive_setup("myservice", cache_dir)

            self.assertTrue(ok)
            self.assertEqual(token, "secret-value")
            self.assertEqual(key, "secret-value")
            self.assertEqual(url, "https://example.com/script")
            self.assertEqual(api_key, "api-key-123")
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0][3], cache_dir / "credgoo.txt")
            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0][0], "myservice")
            self.assertEqual(cached[0][1], "api-key-123")
            self.assertEqual(cached[0][3], cache_dir)

    def test_main_runs_interactive_setup_when_missing_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "svc", "--cache-dir", tmpdir]), \
                    patch.object(mod.sys.stdin, "isatty", return_value=True), \
                    patch.object(mod, "load_credentials", return_value=(None, None, None)), \
                    patch.object(mod, "interactive_setup", return_value=("t", "k", "u", "setup-key", True)), \
                    patch.object(mod, "get_api_key", return_value=None):
                exit_code = mod.main()

            self.assertEqual(exit_code, 0)

    def test_main_setup_flag_without_service_prompts_for_service(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "--setup", "--cache-dir", tmpdir]), \
                    patch.object(mod.sys.stdin, "isatty", return_value=True), \
                    patch.object(mod, "load_credentials", return_value=(None, None, None)), \
                    patch.object(mod, "_prompt_service_name", return_value="svc"), \
                    patch.object(mod, "interactive_setup", return_value=("t", "k", "u", "setup-key", True)):
                exit_code = mod.main()

            self.assertEqual(exit_code, 0)

    def test_main_setup_flag_with_service_still_prompts_for_service(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mod.sys, "argv", ["credgoo", "ignored", "--setup", "--cache-dir", tmpdir]), \
                    patch.object(mod.sys.stdin, "isatty", return_value=True), \
                    patch.object(mod, "load_credentials", return_value=(None, None, None)), \
                    patch.object(mod, "_prompt_service_name", return_value="prompted-svc") as prompt_mock, \
                    patch.object(mod, "interactive_setup", return_value=("t", "k", "u", "setup-key", True)) as setup_mock:
                exit_code = mod.main()

            self.assertEqual(exit_code, 0)
            prompt_mock.assert_called_once_with()
            self.assertEqual(setup_mock.call_args[0][0], "prompted-svc")


if __name__ == "__main__":
    unittest.main()
