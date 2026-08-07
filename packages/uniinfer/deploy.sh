#!/bin/bash
#
# Deploy the uniioai-proxy: git pull -> uv sync (frozen) -> restart.
#
# Provider / extras selection (which optional provider SDKs to install):
#   (no arg)        BASE only — the lean default. Ships the core providers that
#                   need no extra SDK (tu, openai-compat, ollama) plus image
#                   generation (pollinations). Right for a box that only serves
#                   tu; nothing optional is installed, so nothing heavy stays
#                   resident. This is the default so a bare `./deploy.sh` is
#                   always lean and removals stick across redeploys.
#   --base          same as no arg (explicit).
#   --extras A,B,C  base + exactly those extras. Available extras:
#                   anthropic  gemini  mistral  cohere  huggingface  groq  ai21
#   --all-extras    base + every optional provider SDK (opt-in; the old default).

set -e

export PATH="$HOME/.local/bin:$PATH"
ROOT=/home/ubuntu/code/python-openutils
UNIINFER="$ROOT/packages/uniinfer"
EXTRAS_FLAGS=""

usage() {
    sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

case "${1:-}" in
    ""|--base|--no-extras)
        EXTRAS_FLAGS=""
        ;;
    --all-extras)
        EXTRAS_FLAGS="--all-extras"
        ;;
    --extras)
        if [ -z "${2:-}" ]; then
            echo "--extras needs a comma-separated list (e.g. --extras anthropic,groq)"
            exit 1
        fi
        IFS=',' read -ra _xs <<< "$2"
        for _x in "${_xs[@]}"; do
            [ -n "$_x" ] && EXTRAS_FLAGS="$EXTRAS_FLAGS --extra $_x"
        done
        [ -z "$EXTRAS_FLAGS" ] && { echo "--extras list was empty"; exit 1; }
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
esac

echo "Extras selection:${EXTRAS_FLAGS:- (base only — lean default)}"

echo "Pulling latest code..."
cd "$ROOT"
git pull

echo "Syncing dependencies..."
cd "$UNIINFER"
# Frozen: install from the committed uv.lock, never mutate it on the server
# (a bare `uv sync` rewrites a stale uv.lock → blocks the next `git pull`).
if ! uv lock --check; then
    echo "❌ uv.lock is stale — pyproject.toml changed without 'uv lock'."
    echo "   Fix: (cd packages/uniinfer && uv lock) then commit+push uv.lock, and re-run."
    exit 1
fi
uv sync --frozen $EXTRAS_FLAGS

echo "Restarting uniioai-proxy service..."
sudo systemctl restart uniioai-proxy

sleep 3
echo "Checking service status..."
sudo systemctl status uniioai-proxy --no-pager -l | head -5

echo "✅ Update complete!"
