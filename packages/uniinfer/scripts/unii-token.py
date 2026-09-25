#!/usr/bin/env python3
"""unii-token — operator CLI for uniinfer gateway access tokens.

Examples:
    uv run python scripts/unii-token.py mint --name habit
    uv run python scripts/unii-token.py mint --name ci --ttl 7d --provider tu
    uv run python scripts/unii-token.py list
    uv run python scripts/unii-token.py revoke --name ci
    uv run python scripts/unii-token.py prune

The plaintext token is printed to stdout once. Operator details go to stderr.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "uniinfer_token_registry",
    Path(__file__).resolve().parents[1]
    / "uniinfer"
    / "proxy_services"
    / "token_registry.py",
)
registry = importlib.util.module_from_spec(_SPEC)
sys.modules["uniinfer_token_registry"] = registry
_SPEC.loader.exec_module(registry)


def _providers(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    normalized: list[str] = []
    for value in values:
        for provider in value.split(","):
            provider = provider.strip()
            if provider and provider not in normalized:
                normalized.append(provider)
    return normalized


def _record_line(record, now=None) -> str:
    from datetime import datetime

    status = "active"
    if not record.allowlisted:
        status = "orphan"
    elif record.expires_at:
        try:
            deadline = datetime.fromisoformat(record.expires_at.replace("Z", "+00:00"))
            if now is None:
                now = datetime.now(deadline.tzinfo)
            if now >= deadline:
                status = "expired"
        except ValueError:
            status = "invalid-expiry"
    providers = ",".join(record.providers) if record.providers else "all"
    created = record.created_at[:19].replace("T", " ")
    expires = (record.expires_at or "never")[:19].replace("T", " ")
    return (
        f"{record.name:24} {record.sha256[:12]}… {status:9} "
        f"created={created} expires={expires} providers={providers}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the operator CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allowlist",
        help="auth_tokens.allow path (default: UNIINFER_AUTH_TOKENS_FILE)",
    )
    parser.add_argument(
        "--metadata",
        help="metadata JSON path (default: auth_tokens.meta.json beside allowlist)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    mint = commands.add_parser("mint", help="mint a token and print it once")
    mint.add_argument("--name", required=True)
    mint.add_argument("--ttl", default="30d", help="30d, 12h, 1w, or never")
    mint.add_argument(
        "--provider",
        action="append",
        default=[],
        help="restrict to provider/instance alias; repeat or comma-separate",
    )

    commands.add_parser("list", help="list issued tokens without secrets")

    revoke = commands.add_parser("revoke", help="revoke immediately")
    revoke.add_argument("--name")
    revoke.add_argument("--hash", help="exact sha256(token) hex digest")

    commands.add_parser("prune", help="remove expired tokens")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = registry.resolve_paths(args.allowlist, args.metadata)

    try:
        if args.command == "mint":
            registry.parse_ttl(args.ttl)
            token, record = registry.mint_token(
                args.name,
                args.ttl,
                _providers(args.provider),
                allowlist=paths.allowlist,
                metadata=paths.metadata,
            )
            print(
                _record_line(record),
                file=sys.stderr,
            )
            print("Save this token now; it will not be shown again.", file=sys.stderr)
            print(token)
            return 0

        if args.command == "list":
            records = registry.list_tokens(
                allowlist=paths.allowlist, metadata=paths.metadata
            )
            if not records:
                print("No registered tokens.", file=sys.stderr)
                return 0
            for record in records:
                print(_record_line(record))
            return 0

        if args.command == "revoke":
            revoked = registry.revoke_token(
                name=args.name,
                token_hash=args.hash,
                allowlist=paths.allowlist,
                metadata=paths.metadata,
            )
            for record in revoked:
                print(_record_line(record), file=sys.stderr)
            print(f"Revoked {len(revoked)} token(s).", file=sys.stderr)
            return 0

        if args.command == "prune":
            expired = registry.prune_expired(
                allowlist=paths.allowlist, metadata=paths.metadata
            )
            for record in expired:
                print(_record_line(record), file=sys.stderr)
            print(f"Pruned {len(expired)} expired token(s).", file=sys.stderr)
            return 0
    except registry.TokenRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
