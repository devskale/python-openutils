"""Google Apps Script backend — the original credgoo approach.

Setup: user deploys an Apps Script, provides URL + token + encryption key.
Fetch: calls the script with ?service=X&token=Y, gets back XOR-encrypted key.
"""

import base64
import getpass
import logging

import requests

from .base import CredgooBackend

logger = logging.getLogger("credgoo")


def _decrypt_key(encrypted_key, encryption_key):
    """Decrypt the API key using the encryption key (XOR + Base64 + 8-char IV strip)."""
    try:
        decoded = base64.b64decode(encrypted_key).decode("utf-8", errors="replace")
        result = ""
        for i in range(len(decoded)):
            key_char = ord(encryption_key[(i * 7) % len(encryption_key)])
            decoded_char = ord(decoded[i])
            result += chr(decoded_char ^ key_char)
        if len(result) > 8:
            return result[8:]
        logger.warning("Decrypted result too short")
        return result
    except Exception as e:
        logger.error("Decryption error: %s", e)
        return None


class GdriveBackend(CredgooBackend):

    name = "gdrive"

    def setup(self, cache_dir):
        """Interactive setup: prompt for Apps Script URL, token, encryption key."""
        import json
        from pathlib import Path

        print("\ncredgoo: gdrive backend setup\n")
        print("You need a deployed Google Apps Script (see appscript/code.gs).")

        existing_url = None
        existing_token = None
        existing_key = None
        cred_file = cache_dir / "credgoo.txt"
        if cred_file.exists():
            try:
                with open(cred_file) as f:
                    data = json.load(f)
                gdrive = data.get("gdrive", data)
                existing_url = gdrive.get("url")
                existing_token = gdrive.get("token")
                existing_key = gdrive.get("encryption_key")
            except Exception:
                pass

        default_url_text = f" [{existing_url}]" if existing_url else ""
        entered_url = input(f"Google Apps Script URL{default_url_text}: ").strip()
        api_url = entered_url or existing_url
        while not api_url:
            print("URL is required.")
            entered_url = input("Google Apps Script URL: ").strip()
            api_url = entered_url

        default_token_text = " [***hidden***]" if existing_token else ""
        token = input(f"Bearer token{default_token_text}: ").strip() or existing_token
        while not token:
            print("Token is required.")
            token = input("Bearer token: ").strip()

        default_key_text = " [***hidden***]" if existing_key else ""
        encryption_key = input(f"Encryption key{default_key_text}: ").strip() or existing_key
        while not encryption_key:
            print("Encryption key is required.")
            encryption_key = input("Encryption key: ").strip()

        test_service = _prompt_required("Service name to test: ", secret=False)
        logger.info("Testing credentials with service '%s'...", test_service)

        test_key = self.fetch_key(
            test_service,
            {"url": api_url, "token": token, "encryption_key": encryption_key},
        )
        if not test_key:
            print("credgoo: Setup test failed. Credentials were not saved.", file=__import__("sys").stderr)
            return False

        self._store_creds({
            "url": api_url,
            "token": token,
            "encryption_key": encryption_key,
        })
        print("credgoo: Setup complete. Everything looks good.", file=__import__("sys").stderr)
        return True

    def fetch_key(self, service, creds):
        """Call the Apps Script with service + token, decrypt the response."""
        url = creds.get("url")
        bearer_token = creds.get("token")
        encryption_key = creds.get("encryption_key")

        if not url or not bearer_token or not encryption_key:
            logger.error("gdrive backend requires url, token, and encryption_key in credentials.")
            return None

        params = {"service": service, "token": bearer_token}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    encrypted_key = data.get("encryptedKey")
                    if encrypted_key:
                        return _decrypt_key(encrypted_key, encryption_key)
                    logger.error("No encrypted key in response")
                else:
                    logger.error("%s", data.get("message", "Unknown error"))
            else:
                logger.error("Failed to retrieve key (HTTP %d)", response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error("Request error: %s", e)
        return None


def _prompt_required(prompt_text, secret=False):
    while True:
        if secret:
            value = getpass.getpass(prompt_text).strip()
        else:
            value = input(prompt_text).strip()
        if value:
            return value
        print("Value is required.")
