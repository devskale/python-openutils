"""Airtable backend — store API keys in an Airtable base.

Setup: user provides Personal Access Token only.
       CLI auto-creates base + table + columns.
Fetch: filters the table by service name, returns the key field.
"""

import hashlib
import logging
import sys

import requests

from .base import CredgooBackend

logger = logging.getLogger("credgoo")

AIRTABLE_API_BASE = "https://api.airtable.com/v0"
AIRTABLE_META_BASE = "https://api.airtable.com/v0/meta"
BASE_NAME = "credgoo"
TABLE_NAME = "keys"
FIELDS = [
    {"name": "service", "type": "singleLineText"},
    {"name": "key", "type": "singleLineText"},
    {"name": "note", "type": "singleLineText"},
]


def _headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _derive_key_from_pat(pat):
    """Derive a 32-char cache-encryption key from an Airtable PAT via SHA-256."""
    return hashlib.sha256(pat.encode('utf-8')).hexdigest()[:32]


class AirtableClient:
    """Resolved handle to one Airtable keys table.

    Holds the credentials, builds table URLs, and absorbs the plumbing that was
    duplicated across the backend's record methods: the credential guard, the
    transport-error handling, and the offset-pagination loop. Constructed per
    operation from the backend's creds.
    """

    def __init__(self, creds):
        self.token = creds.get("airtable_token")
        self.base_id = creds.get("airtable_base")
        self.table = creds.get("airtable_table", TABLE_NAME)

    @property
    def configured(self):
        return bool(self.token and self.base_id)

    def _table_url(self, *parts):
        url = f"{AIRTABLE_API_BASE}/{self.base_id}/{self.table}"
        return f"{url}/{'/'.join(parts)}" if parts else url

    def _request(self, method, url, **kwargs):
        """Issue a request, returning the response or None on transport error."""
        kwargs.setdefault("timeout", 10)
        try:
            return getattr(requests, method.lower())(url, headers=_headers(self.token), **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable %s request error: %s", method, e)
            return None

    def find_record(self, service):
        """One record matching *service* (filterByFormula), or None."""
        resp = self._request(
            "GET", self._table_url(),
            params={"filterByFormula": f"{{service}} = '{service}'", "maxRecords": 1})
        if resp and resp.status_code == 200:
            records = resp.json().get("records", [])
            if records:
                return records[0]
        return None

    def get_field(self, service, field):
        """A single *field* for *service*, or None."""
        resp = self._request(
            "GET", self._table_url(),
            params={"filterByFormula": f"{{service}} = '{service}'", "maxRecords": 1, "fields[]": [field]})
        if not resp or resp.status_code != 200:
            if resp:
                logger.error("Airtable API error (HTTP %d): %s", resp.status_code, resp.text)
            return None
        records = resp.json().get("records", [])
        if not records:
            logger.error("Service '%s' not found in Airtable", service)
            return None
        return records[0].get("fields", {}).get(field)

    def upsert(self, service, api_key):
        """Insert or update the key for *service*. Returns True on success."""
        existing = self.find_record(service)
        if existing:
            resp = self._request("PATCH", self._table_url(existing["id"]),
                                 json={"fields": {"key": api_key}})
        else:
            resp = self._request("POST", self._table_url(),
                                 json={"fields": {"service": service, "key": api_key}})
        if resp and resp.status_code in (200, 201):
            return True
        if resp:
            logger.error("Airtable write error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def patch_record(self, record_id, fields):
        """Patch a record's fields. Returns True on success."""
        resp = self._request("PATCH", self._table_url(record_id), json={"fields": fields})
        if resp and resp.status_code == 200:
            return True
        if resp:
            logger.error("Airtable patch error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def delete_record(self, record_id):
        """Delete one record. Returns True on success."""
        resp = self._request("DELETE", self._table_url(record_id))
        if resp and resp.status_code == 200:
            return True
        if resp:
            logger.error("Airtable delete error (HTTP %d): %s", resp.status_code, resp.text)
        return False

    def iter_records(self, fields):
        """Yield records page by page, following Airtable's offset cursor."""
        offset = None
        while True:
            params = {"fields[]": fields}
            if offset:
                params["offset"] = offset
            resp = self._request("GET", self._table_url(), params=params)
            if not resp or resp.status_code != 200:
                if resp:
                    logger.error("Airtable list error (HTTP %d): %s", resp.status_code, resp.text)
                return
            data = resp.json()
            yield from data.get("records", [])
            offset = data.get("offset")
            if not offset:
                return

    def delete_records(self, record_ids):
        """Batch-delete records (Airtable caps at 10 per request)."""
        for i in range(0, len(record_ids), 10):
            self._request("DELETE", self._table_url(), params={"records[]": record_ids[i:i + 10]})


class AirtableBackend(CredgooBackend):
    """Airtable backend. Record operations delegate to AirtableClient; setup and
    base/table provisioning run against the meta API directly."""

    name = "airtable"

    cache_integrity = True

    def cache_key(self, creds):
        """Derive the cache key from the PAT (no PAT -> no caching)."""
        pat = creds.get("airtable_token")
        return _derive_key_from_pat(pat) if pat else None

    def _client(self, creds):
        """Build a configured AirtableClient, or None if creds are incomplete."""
        client = AirtableClient(creds)
        if not client.configured:
            logger.error("airtable backend requires airtable_token and airtable_base.")
            return None
        return client

    def fetch_key(self, service, creds):
        """Query Airtable for a service, return the key field."""
        client = self._client(creds)
        return client.get_field(service, "key") if client else None

    def add_key(self, service, api_key, creds):
        """Add or update an API key for a service."""
        client = self._client(creds)
        if not client:
            return False
        if client.upsert(service, api_key):
            logger.info("Key for '%s' saved to Airtable.", service)
            return True
        return False

    def delete_key(self, service, creds):
        """Delete a key by service name."""
        client = self._client(creds)
        if not client:
            return False
        record = client.find_record(service)
        if not record:
            logger.error("Service '%s' not found in Airtable.", service)
            return False
        if client.delete_record(record["id"]):
            logger.info("Key for '%s' deleted from Airtable.", service)
            return True
        return False

    def clear_key(self, service, creds):
        """Blank a key and add a 'cleared at' timestamp in the note column."""
        from datetime import datetime, timezone

        client = self._client(creds)
        if not client:
            return False
        record = client.find_record(service)
        if not record:
            logger.error("Service '%s' not found in Airtable.", service)
            return False
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if client.patch_record(record["id"], {"key": "", "note": f"cleared {now}"}):
            logger.info("Key for '%s' cleared at %s.", service, now)
            return True
        return False

    def list_keys(self, creds):
        """List all service names."""
        client = self._client(creds)
        if not client:
            return []
        return sorted(r["fields"].get("service", "") for r in client.iter_records(["service"]))

    def dedupe_keys(self, creds):
        """Remove duplicate service entries, keeping the first. Returns (kept, removed)."""
        client = self._client(creds)
        if not client:
            return 0, 0
        records = list(client.iter_records(["service"]))
        seen, to_delete = set(), []
        for r in records:
            svc = r["fields"].get("service", "")
            if not svc or svc in seen:
                to_delete.append(r["id"])
            else:
                seen.add(svc)
        if to_delete:
            client.delete_records(to_delete)
        return len(records) - len(to_delete), len(to_delete)

    def setup(self, cache_dir):
        """Interactive setup: prompt for PAT, auto-create base + table."""
        import json
        from pathlib import Path

        print("\ncredgoo: airtable backend setup\n")
        print("You need an Airtable Personal Access Token with these scopes:")
        print("  - data.records:read")
        print("  - data.records:write")
        print("  - schema.bases:write")
        print("Create one at: https://airtable.com/create/tokens\n")

        existing_token = None
        existing_base = None
        cred_file = cache_dir / "credgoo.txt"
        if cred_file.exists():
            try:
                with open(cred_file) as f:
                    data = json.load(f)
                at = data.get("airtable", {})
                existing_token = at.get("airtable_token")
                existing_base = at.get("airtable_base")
            except Exception:
                pass

        default_token_text = " [***hidden***]" if existing_token else ""
        token = input(f"Airtable Personal Access Token{default_token_text}: ").strip() or existing_token
        while not token:
            print("Token is required.")
            token = input("Airtable Personal Access Token: ").strip()

        base_id = existing_base
        if not base_id:
            logger.info("Looking for existing '%s' base...", BASE_NAME)
            base_id = self._find_base(token)

        if base_id:
            print(f"Found existing base: {base_id}", file=sys.stderr)
        else:
            print(f"Creating '{BASE_NAME}' base...", file=sys.stderr)
            base_id = self._create_base(token)
            if not base_id:
                print("credgoo: Failed to create base. Check token scopes.", file=sys.stderr)
                return False
            print(f"Created base: {base_id}", file=sys.stderr)

        logger.info("Ensuring table '%s' with correct columns...", TABLE_NAME)
        if not self._ensure_table(token, base_id):
            print("credgoo: Failed to set up table.", file=sys.stderr)
            return False

        self._store_creds({
            "airtable_token": token,
            "airtable_base": base_id,
            "airtable_table": TABLE_NAME,
        })

        print(f"\ncredgoo: Setup complete! ✅")
        print(f"  Base: https://airtable.com/{base_id}")
        print(f"  Add keys: column 'service' = name, column 'key' = secret")
        print(f"  Or use: credgoo --add <service> <key>\n")
        return True

    # ---- Auto-setup helpers (meta API — one-time provisioning) ----

    def _find_base(self, token):
        """Find an existing base named BASE_NAME. Returns base_id or None."""
        try:
            resp = requests.get(AIRTABLE_META_BASE + "/bases", headers=_headers(token), timeout=10)
            if resp.status_code == 200:
                for base in resp.json().get("bases", []):
                    if base.get("name") == BASE_NAME:
                        return base["id"]
        except requests.exceptions.RequestException as e:
            logger.error("Airtable list bases error: %s", e)
        return None

    def _create_base(self, token):
        """Create a new base with the keys table. Returns base_id or None."""
        try:
            resp = requests.post(
                AIRTABLE_META_BASE + "/bases",
                headers=_headers(token),
                json={
                    "name": BASE_NAME,
                    "tables": [{
                        "name": TABLE_NAME,
                        "fields": FIELDS,
                    }],
                },
                timeout=15,
            )
            if resp.status_code in (200, 201):
                return resp.json().get("id") or resp.json().get("base", {}).get("id")
            logger.error("Create base error (HTTP %d): %s", resp.status_code, resp.text)
        except requests.exceptions.RequestException as e:
            logger.error("Create base error: %s", e)
        return None

    def _ensure_table(self, token, base_id):
        """Ensure the keys table exists with correct columns."""
        try:
            resp = requests.get(
                f"{AIRTABLE_META_BASE}/bases/{base_id}/tables",
                headers=_headers(token),
                timeout=10,
            )
            if resp.status_code != 200:
                logger.error("List tables error (HTTP %d): %s", resp.status_code, resp.text)
                return False

            tables = resp.json().get("tables", [])
            for table in tables:
                if table["name"] == TABLE_NAME:
                    logger.info("Table '%s' already exists.", TABLE_NAME)
                    return True

            resp = requests.post(
                f"{AIRTABLE_META_BASE}/bases/{base_id}/tables",
                headers=_headers(token),
                json={
                    "name": TABLE_NAME,
                    "fields": FIELDS,
                },
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logger.info("Created table '%s'.", TABLE_NAME)
                return True
            logger.error("Create table error (HTTP %d): %s", resp.status_code, resp.text)
        except requests.exceptions.RequestException as e:
            logger.error("Ensure table error: %s", e)
        return False
