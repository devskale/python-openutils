"""Mitschnitt-Proxy between a client and this repo's own proxy (or any upstream).

Captures the exact request payloads and responses while relaying them — the
capture-first alternative to reconstructing requests from pi session files
(session format != wire format; reconstruction has burned whole debug cycles).

Usage:
  1. python3 scripts/debug-capture-proxy.py &        # listens on 127.0.0.1:8199
  2. Point the client at it — for pi: providers.unii.baseUrl = http://127.0.0.1:8199/v1
     in ~/.pi/agent/models.json (RESTORE afterwards; authHeader: true passes the
     real token through).
  3. Reproduce. Requests + responses land in /tmp/captured.jsonl.
  4. Restore models.json. Bisect the captured payload against the live gateway.

Always stop the proxy and restore the client config when done.
"""
import http.server
import json
import sys
import threading
import urllib.request

UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "https://uniinfer.skale.dev"
LOG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/captured.jsonl"
lock = threading.Lock()


def record(entry):
    with lock, open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        # Relay verbatim, Authorization INCLUDED (the client's real token).
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)
        try:
            up = urllib.request.urlopen(req, timeout=300)
            data, code, ct = up.read(), up.status, up.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e:
            data, code, ct = e.read(), e.code, e.headers.get("content-type", "application/json")
        record({"path": self.path, "req": parsed, "resp_status": code,
                "resp": data.decode("utf-8", "replace")})
        self.send_response(code)
        self.send_header("content-type", ct)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"capture proxy: 127.0.0.1:8199 -> {UPSTREAM}, log: {LOG}")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8199), Handler).serve_forever()
