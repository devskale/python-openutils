#!/bin/bash

set -e

export PATH="$HOME/.local/bin:$PATH"
ROOT=/home/ubuntu/code/python-openutils
UNIINFER="$ROOT/packages/uniinfer"
EXTRAS=""

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all-extras    Install all optional provider dependencies (default)"
    echo "  --no-extras     Skip optional provider dependencies"
    echo "  -h, --help      Show this help"
    exit 0
}

if [ $# -eq 0 ]; then
    EXTRAS="--all-extras"
else
    case "$1" in
        --all-extras) EXTRAS="--all-extras" ;;
        --no-extras)  EXTRAS="" ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
fi

echo "Pulling latest code..."
cd "$ROOT"
git pull

echo "Syncing dependencies...${EXTRAS:+ ($EXTRAS)}"
cd "$UNIINFER"
uv sync $EXTRAS

echo "Restarting uniioai-proxy service..."
sudo systemctl restart uniioai-proxy

sleep 3
echo "Checking service status..."
sudo systemctl status uniioai-proxy --no-pager -l | head -5

echo "✅ Update complete!"
