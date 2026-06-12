"""Airtable backend — store API keys in an Airtable base.

Setup: user provides Personal Access Token only.
       CLI auto-creates base + table + columns.
Fetch: filters the table by service name, returns the key field.
"""

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


class AirtableBackend(CredgooBackend):

    name = "airtable"

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

    def fetch_key(self, service, creds):
        """Query Airtable for a specific service, return the key field."""
        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)

        if not token or not base_id:
            logger.error("airtable backend requires airtable_token and airtable_base.")
            return None

        url = f"{AIRTABLE_API_BASE}/{base_id}/{table}"
        params = {
            "filterByFormula": f"{{service}} = '{service}'",
            "maxRecords": 1,
            "fields[]": ["key"],
        }
        try:
            resp = requests.get(url, headers=_headers(token), params=params, timeout=10)
            if resp.status_code != 200:
                logger.error("Airtable API error (HTTP %d): %s", resp.status_code, resp.text)
                return None
            records = resp.json().get("records", [])
            if not records:
                logger.error("Service '%s' not found in Airtable", service)
                return None
            return records[0].get("fields", {}).get("key")
        except requests.exceptions.RequestException as e:
            logger.error("Airtable request error: %s", e)
        return None

    def add_key(self, service, api_key, creds):
        """Add or update an API key for a service."""
        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)

        if not token or not base_id:
            logger.error("airtable backend requires airtable_token and airtable_base.")
            return False

        existing = self._find_record(token, base_id, table, service)
        try:
            if existing:
                record_id = existing["id"]
                resp = requests.patch(
                    f"{AIRTABLE_API_BASE}/{base_id}/{table}/{record_id}",
                    headers=_headers(token),
                    json={"fields": {"key": api_key}},
                    timeout=10,
                )
            else:
                resp = requests.post(
                    f"{AIRTABLE_API_BASE}/{base_id}/{table}",
                    headers=_headers(token),
                    json={"fields": {"service": service, "key": api_key}},
                    timeout=10,
                )
            if resp.status_code in (200, 201):
                logger.info("Key for '%s' saved to Airtable.", service)
                return True
            logger.error("Airtable write error (HTTP %d): %s", resp.status_code, resp.text)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable write error: %s", e)
        return False

    def delete_key(self, service, creds):
        """Delete a key by service name. Returns True if deleted."""
        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)

        if not token or not base_id:
            logger.error("airtable backend requires airtable_token and airtable_base.")
            return False

        record = self._find_record(token, base_id, table, service)
        if not record:
            logger.error("Service '%s' not found in Airtable.", service)
            return False
        try:
            resp = requests.delete(
                f"{AIRTABLE_API_BASE}/{base_id}/{table}/{record['id']}",
                headers=_headers(token),
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Key for '%s' deleted from Airtable.", service)
                return True
            logger.error("Airtable delete error (HTTP %d): %s", resp.status_code, resp.text)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable delete error: %s", e)
        return False

    def clear_key(self, service, creds):
        """Blank a key and add 'cleared at' timestamp in the note column."""
        from datetime import datetime, timezone

        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)

        if not token or not base_id:
            logger.error("airtable backend requires airtable_token and airtable_base.")
            return False

        record = self._find_record(token, base_id, table, service)
        if not record:
            logger.error("Service '%s' not found in Airtable.", service)
            return False

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = requests.patch(
                f"{AIRTABLE_API_BASE}/{base_id}/{table}/{record['id']}",
                headers=_headers(token),
                json={"fields": {"key": "", "note": f"cleared {now}"}},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Key for '%s' cleared at %s.", service, now)
                return True
            logger.error("Airtable clear error (HTTP %d): %s", resp.status_code, resp.text)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable clear error: %s", e)
        return False

    # ---- Auto-setup helpers ----

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

    def _find_record(self, token, base_id, table, service):
        """Find a record by service name. Returns record dict or None."""
        try:
            resp = requests.get(
                f"{AIRTABLE_API_BASE}/{base_id}/{table}",
                headers=_headers(token),
                params={"filterByFormula": f"{{service}} = '{service}'", "maxRecords": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                records = resp.json().get("records", [])
                if records:
                    return records[0]
        except requests.exceptions.RequestException:
            pass
        return None

    def list_keys(self, creds):
        """List all service names. Returns list of str."""
        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)
        if not token or not base_id:
            return []
        try:
            all_records = []
            offset = None
            while True:
                params = {"fields[]": ["service"]}
                if offset:
                    params["offset"] = offset
                resp = requests.get(
                    f"{AIRTABLE_API_BASE}/{base_id}/{table}",
                    headers=_headers(token),
                    params=params,
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.error("Airtable list error (HTTP %d): %s", resp.status_code, resp.text)
                    break
                data = resp.json()
                all_records.extend(data.get("records", []))
                offset = data.get("offset")
                if not offset:
                    break
            return sorted(r["fields"].get("service", "") for r in all_records)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable list error: %s", e)
        return []

    def dedupe_keys(self, creds):
        """Remove duplicate service entries, keeping the first. Returns (kept, removed) counts."""
        token = creds.get("airtable_token")
        base_id = creds.get("airtable_base")
        table = creds.get("airtable_table", TABLE_NAME)
        if not token or not base_id:
            return 0, 0

        try:
            all_records = []
            offset = None
            while True:
                params = {"fields[]": ["service"]}
                if offset:
                    params["offset"] = offset
                resp = requests.get(
                    f"{AIRTABLE_API_BASE}/{base_id}/{table}",
                    headers=_headers(token),
                    params=params,
                    timeout=10,
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                all_records.extend(data.get("records", []))
                offset = data.get("offset")
                if not offset:
                    break

            seen = {}
            to_delete = []
            for r in all_records:
                svc = r["fields"].get("service", "")
                if not svc or svc in seen:
                    to_delete.append(r["id"])
                else:
                    seen[svc] = r["id"]

            if not to_delete:
                return len(all_records), 0

            for i in range(0, len(to_delete), 10):
                batch = to_delete[i:i + 10]
                requests.delete(
                    f"{AIRTABLE_API_BASE}/{base_id}/{table}",
                    headers=_headers(token),
                    params={"records[]": batch},
                    timeout=10,
                )
            return len(all_records) - len(to_delete), len(to_delete)
        except requests.exceptions.RequestException as e:
            logger.error("Airtable dedupe error: %s", e)
        return 0, 0
