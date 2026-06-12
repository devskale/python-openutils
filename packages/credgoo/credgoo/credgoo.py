import logging
import requests
import base64
import argparse
import os
import sys
import json
import time
import getpass
import secrets as _secrets
import string
import http.server
import webbrowser
import urllib.parse
from pathlib import Path
from importlib.metadata import version

logger = logging.getLogger("credgoo")

OAUTH_CLIENT_ID = "REDACTED"
OAUTH_CLIENT_SECRET = "REDACTED"
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
TEMPLATE_SHEET_ID = "19IJncFxaZcitwiwcvTjCWsvXfrFSLXJ7pPCmz3Sw2Zc"


# ---- Encryption (for local cache) ----

def encrypt_local_key(api_key, encryption_key):
    """Encrypt the API key for local caching using XOR and Base64."""
    try:
        encrypted_bytes = bytearray()
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        for i, char_code in enumerate(api_key.encode('utf-8')):
            key_char_code = key_bytes[i % key_len]
            encrypted_bytes.append(char_code ^ key_char_code)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        logger.error("Local encryption error: %s", e)
        return None


def decrypt_local_key(encrypted_api_key, encryption_key):
    """Decrypt the locally cached API key using XOR."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_api_key)
        decrypted_bytes = bytearray()
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        for i, byte_val in enumerate(encrypted_bytes):
            key_char_code = key_bytes[i % key_len]
            decrypted_bytes.append(byte_val ^ key_char_code)
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.error("Local decryption error: %s", e)
        return None


# ---- OAuth ----

def oauth_flow():
    """Run OAuth2 desktop flow. Returns (access_token, refresh_token) or (None, None)."""
    auth_code = None
    state_token = _secrets.token_urlsafe(16)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            if params.get("state", [None])[0] != state_token:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state")
                return
            auth_code = params.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if auth_code:
                self.wfile.write(b"<html><body><h1>credgoo: Authorized! You can close this tab.</h1></body></html>")
            else:
                self.wfile.write(b"<html><body><h1>credgoo: Authorization failed.</h1></body></html>")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    redirect_uri = f"http://localhost:{port}"
    server.timeout = 120

    params = urllib.parse.urlencode({
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "state": state_token,
        "access_type": "offline",
        "prompt": "consent",
    })
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{params}"

    print("Opening browser for Google authorization...")
    webbrowser.open(auth_url)

    server.handle_request()
    server.server_close()

    if not auth_code:
        return None, None

    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=15)

    data = resp.json()
    if "error" in data:
        logger.error("Token exchange failed: %s", data.get("error"))
        return None, None

    return data.get("access_token"), data.get("refresh_token")


def get_access_token(refresh_token):
    """Get a fresh access token using the refresh token."""
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }, timeout=15)
    data = resp.json()
    if "access_token" in data:
        return data["access_token"]
    logger.error("Token refresh failed: %s", data.get("error"))
    return None


def copy_template_sheet(access_token):
    """Copy the template sheet to the user's Google Drive. Returns new sheet ID."""
    resp = requests.post(
        f"https://www.googleapis.com/drive/v3/files/{TEMPLATE_SHEET_ID}/copy",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"name": "API Keys (credgoo)"},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json()["id"]
    logger.error("Failed to copy template: %s", resp.text)
    return None


def get_api_key_from_sheets(service, access_token, sheet_id):
    """Retrieve an API key directly from Google Sheets API."""
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/keys!A:B"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        if resp.status_code != 200:
            logger.error("Sheets API error (HTTP %d): %s", resp.status_code, resp.text)
            return None
        values = resp.json().get("values", [])
        for row in values:
            if row and row[0] == service:
                return row[1] if len(row) > 1 else None
        logger.error("Service '%s' not found in sheet", service)
    except requests.exceptions.RequestException as e:
        logger.error("Sheets API error: %s", e)
    return None


# ---- Caching ----

def cache_api_key(service, api_key, encryption_key, cache_dir):
    """Store encrypted API key in service-specific cache file."""
    if not encryption_key:
        logger.warning("Cannot cache API key without an encryption key.")
        return

    encrypted_key_for_cache = encrypt_local_key(api_key, encryption_key)
    if not encrypted_key_for_cache:
        logger.warning("Failed to encrypt API key for caching.")
        return

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "service": service,
            "api_key": encrypted_key_for_cache,
            "timestamp": str(int(time.time()))
        }

        cache_file = cache_dir / 'api_keys.json'

        existing_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    existing_cache = json.load(f)
            except json.JSONDecodeError:
                existing_cache = {}

        existing_cache[service] = cache_data

        with open(cache_file, 'w') as f:
            json.dump(existing_cache, f, indent=2)

        os.chmod(cache_file, 0o600)
        logger.info("API key for %s cached (encrypted)", service)
    except Exception as e:
        logger.warning("Failed to cache API key: %s", e)


def get_cached_api_key(service, encryption_key, cache_dir):
    """Retrieve and decrypt API key for specific service from cache."""
    if not encryption_key:
        logger.warning("Cannot decrypt cached key without an encryption key.")
        return None

    cache_file = cache_dir / 'api_keys.json'

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)

        if service in cache:
            encrypted_cached_key = cache[service].get("api_key")
            if encrypted_cached_key:
                decrypted_key = decrypt_local_key(
                    encrypted_cached_key, encryption_key)
                if decrypted_key:
                    logger.info("Using cached key for %s", service)
                    return decrypted_key
                else:
                    logger.info("Retrieving key for %s.", service)
            else:
                logger.warning("No 'api_key' field in cache for %s.", service)

    except Exception as e:
        logger.warning("Failed to read cached API key: %s", e)

    return None


# ---- Credential storage ----

def store_credentials(creds, cred_file):
    """Store credentials dict to file."""
    try:
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cred_file, 'w') as f:
            json.dump(creds, f)
        os.chmod(cred_file, 0o600)
        logger.info("Credentials saved to %s", cred_file)
    except Exception as e:
        logger.warning("Failed to store credentials: %s", e)


def load_credentials(cred_file):
    """Load credentials as a dict. Returns empty dict if missing."""
    try:
        if cred_file.exists():
            with open(cred_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning("Failed to load credentials: %s", e)
        return {}


# ---- Main public API ----

def get_api_key(service, cache_dir=None, no_cache=False):
    """
    Get API key for a service. Uses OAuth + Sheets API.
    Returns plaintext key (str) or None.
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    refresh_token = creds.get("refresh_token")
    sheet_id = creds.get("sheet_id")
    enc_key = creds.get("encryption_key")

    if not refresh_token or not sheet_id:
        logger.error("No credentials configured. Run 'credgoo --setup-backend'.")
        return None

    if not no_cache and enc_key:
        cached_key = get_cached_api_key(service, enc_key, cache_dir)
        if cached_key:
            return cached_key

    access_token = get_access_token(refresh_token)
    if not access_token:
        return None

    api_key = get_api_key_from_sheets(service, access_token, sheet_id)
    if api_key and not no_cache and enc_key:
        cache_api_key(service, api_key, enc_key, cache_dir)
    return api_key


# ---- Setup ----

def setup_backend(cache_dir):
    """Guided setup: OAuth authorize → copy template sheet → test → save."""
    print("\ncredgoo: Backend setup\n")

    access_token, refresh_token = oauth_flow()
    if not refresh_token:
        print("credgoo: Authorization failed.", file=sys.stderr)
        return False

    print("Copying template sheet to your Google Drive...")
    sheet_id = copy_template_sheet(access_token)
    if not sheet_id:
        print("credgoo: Failed to copy template sheet.", file=sys.stderr)
        return False

    encryption_key = ''.join(_secrets.choice(string.ascii_letters + string.digits) for _ in range(32))

    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    print(f"\nDone! Add your API keys here:")
    print(f"  {sheet_url}")
    print(f"\n  Column A = service name (e.g. openai, groq, gemini)")
    print(f"  Column B = API key")

    cred_file = cache_dir / 'credgoo.txt'
    store_credentials({
        "mode": "oauth",
        "refresh_token": refresh_token,
        "sheet_id": sheet_id,
        "encryption_key": encryption_key,
    }, cred_file)

    api_key = get_api_key_from_sheets("demo", access_token, sheet_id)
    if api_key:
        cache_api_key("demo", api_key, encryption_key, cache_dir)
        print(f"\ncredgoo: Setup complete. Tested with 'demo' — everything works.")
        print(f"⚠️  Replace the demo key with your real API keys!")
    else:
        print(f"\ncredgoo: Setup complete. Add keys to your sheet, then run 'credgoo <service>'.")

    print(f"Sheet: {sheet_url}\n")
    return True


def _prompt_required(prompt_text, secret=False):
    """Prompt for a required value until non-empty input is given."""
    while True:
        if secret:
            value = getpass.getpass(prompt_text).strip()
        else:
            value = input(prompt_text).strip()
        if value:
            return value
        print("Value is required.")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve API keys securely from Google Sheets")
    parser.add_argument(
        "service", nargs="?", help="Service name to retrieve the API key for")
    parser.add_argument("--setup-backend", action="store_true",
                        help="Guided setup: authorize Google, copy template sheet")
    parser.add_argument(
        "--cache-dir", help="Directory to store cached API keys (default: ~/.config/api_keys/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass cache and force retrieval from source")
    parser.add_argument("--update", action="store_true",
                        help="Force fresh fetch and update cache")
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + version('credgoo'),
                        help="Show program's version number and exit")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stderr)

    if not args.setup_backend and not args.service:
        parser.error("the following arguments are required: service")

    cache_dir = Path(args.cache_dir) if args.cache_dir else (Path.home() / '.config' / 'api_keys')

    if args.setup_backend:
        if not sys.stdin.isatty():
            print("Error: Backend setup requires a TTY.", file=sys.stderr)
            return 1
        return 0 if setup_backend(cache_dir) else 1

    force_no_cache = args.no_cache or args.update

    current_cached_key = None
    if args.update:
        current_cached_key = get_api_key(args.service, cache_dir=cache_dir, no_cache=False)
        if current_cached_key:
            print(f"credgoo: Found cached key for {args.service}.", file=sys.stderr)
        else:
            print(f"credgoo: No cached key found for {args.service}.", file=sys.stderr)

    api_key = get_api_key(args.service, cache_dir=cache_dir, no_cache=force_no_cache)

    if api_key and api_key == "replace-with-your-api-key":
        print(f"⚠️  You still have the demo key for '{args.service}'. Replace it in your sheet:", file=sys.stderr)
        creds = load_credentials(cache_dir / 'credgoo.txt')
        sheet_id = creds.get("sheet_id", "")
        if sheet_id:
            print(f"   https://docs.google.com/spreadsheets/d/{sheet_id}", file=sys.stderr)
        return 1

    if args.update and api_key and current_cached_key:
        if api_key != current_cached_key:
            print(f"credgoo: Online key for {args.service} has changed. Updating cache.", file=sys.stderr)
        else:
            print(f"credgoo: Online key for {args.service} is up to date.", file=sys.stderr)
    elif args.update and api_key and not current_cached_key:
        print(f"credgoo: Fetched online key for {args.service}. No previous cache.", file=sys.stderr)
    elif args.update and not api_key:
        print(f"credgoo: Failed to fetch online key for {args.service}.", file=sys.stderr)

    if api_key:
        print(api_key)
        return 0
    else:
        print("credgoo: Failed to retrieve API key", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
