"""Cloudflare Worker backend — keys served by a self-hosted Worker + D1.

Setup: user provides the worker URL + bearer token (worker deployed from
coreservices/cloudflare/credgoo). Keys are AES-GCM encrypted at rest by the
worker; this client speaks its small JSON API. Because ``service`` is the
primary key upstream, duplicates cannot exist and ``dedupe_keys`` stays
unsupported on purpose.
"""

import hashlib
import logging
import sys

import requests

from .base import CredgooBackend

logger = logging.getLogger("credgoo")


def _derive_key_from_token(token):
    """Derive a 32-char cache-encryption key from the worker token via SHA-256."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()[:32]


def _svc_path(service):
    """URL-encode a service name for the path segment."""
    return requests.utils.quote(service, safe='')


class CfworkerBackend(CredgooBackend):
    """Self-hosted credgoo Worker (D1 storage, AES-GCM at rest)."""

    name = "cfworker"

    cache_integrity = True

    def cache_key(self, creds):
        """Derive the cache key from the token (no token -> no caching)."""
        token = creds.get("token")
        return _derive_key_from_token(token) if token else None

    def _url(self, creds, path):
        return f"{(creds.get('url') or '').rstrip('/')}{path}"

    def _request(self, method, creds, path, **kwargs):
        """Issue a request, returning the response or None on transport error."""
        kwargs.setdefault("timeout", 10)
        headers = {
            "Authorization": f"Bearer {creds.get('token')}",
            "Content-Type": "application/json",
        }
        try:
            return getattr(requests, method.lower())(self._url(creds, path), headers=headers, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error("cfworker %s request error: %s", method, e)
            return None

    def fetch_key(self, service, creds):
        """GET the service row; 404 and cleared keys both count as missing."""
        if not creds.get("url") or not creds.get("token"):
            logger.error("cfworker backend requires url and token in credentials.")
            return None
        resp = self._request("GET", creds, f"/v1/keys/{_svc_path(service)}")
        if not resp:
            return None
        if resp.status_code == 200:
            return resp.json().get("key") or None
        if resp.status_code != 404:
            logger.error("cfworker fetch error (HTTP %d): %s", resp.status_code, resp.text)
        return None

    def add_key(self, service, api_key, creds):
        """Upsert the key for a service."""
        resp = self._request("PUT", creds, f"/v1/keys/{_svc_path(service)}", json={"key": api_key})
        if resp and resp.status_code == 200:
            return True
        if resp:
            logger.error("cfworker write error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def delete_key(self, service, creds):
        """Delete a key by service name."""
        resp = self._request("DELETE", creds, f"/v1/keys/{_svc_path(service)}")
        if resp and resp.status_code == 200:
            return True
        if resp and resp.status_code == 404:
            logger.error("Service '%s' not found in cfworker.", service)
        elif resp:
            logger.error("cfworker delete error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def clear_key(self, service, creds):
        """Blank a key and stamp cleared_at upstream."""
        resp = self._request("POST", creds, f"/v1/keys/{_svc_path(service)}/clear")
        if resp and resp.status_code == 200:
            return True
        if resp and resp.status_code == 404:
            logger.error("Service '%s' not found in cfworker.", service)
        elif resp:
            logger.error("cfworker clear error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def list_keys(self, creds):
        """List all service names."""
        resp = self._request("GET", creds, "/v1/keys")
        if not resp or resp.status_code != 200:
            if resp:
                logger.error("cfworker list error (HTTP %d): %s", resp.status_code, resp.text)
            return []
        return sorted(resp.json().get("services", []))

    def setup(self, cache_dir):
        """Interactive setup: prompt for worker URL + token, probe, save."""
        import json

        print("\ncredgoo: cfworker backend setup\n")
        print("You need a deployed credgoo Worker (see README in")
        print("coreservices/cloudflare/credgoo).")

        existing_url = None
        existing_token = None
        cred_file = cache_dir / "credgoo.txt"
        if cred_file.exists():
            try:
                with open(cred_file) as f:
                    data = json.load(f)
                cfw = data.get("cfworker", {})
                existing_url = cfw.get("url")
                existing_token = cfw.get("token")
            except Exception:
                pass

        default_url_text = f" [{existing_url}]" if existing_url else ""
        entered_url = input(f"Worker URL{default_url_text}: ").strip()
        url = entered_url or existing_url
        while not url:
            print("URL is required.")
            url = input("Worker URL: ").strip()

        default_token_text = " [***hidden***]" if existing_token else ""
        token = input(f"Bearer token{default_token_text}: ").strip() or existing_token
        while not token:
            print("Token is required.")
            token = input("Bearer token: ").strip()

        creds = {"url": url, "token": token}
        probe = self._request("GET", creds, "/")
        if not probe or probe.status_code != 200:
            print("credgoo: Setup test failed. Credentials were not saved.", file=sys.stderr)
            return False

        self._store_creds(creds)
        print("credgoo: Setup complete. Everything looks good.", file=sys.stderr)
        return True
