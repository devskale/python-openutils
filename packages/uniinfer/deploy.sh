#!/bin/bash

set -e

export PATH="$HOME/.local/bin:$PATH"
ROOT=/home/ubuntu/code/python-openutils
UNIINFER="$ROOT/packages/uniinfer"

echo "Pulling latest code..."
cd "$ROOT"
git pull

echo "Syncing dependencies..."
cd "$UNIINFER"
uv sync

echo "Restarting uniioai-proxy service..."
sudo systemctl restart uniioai-proxy

sleep 3
echo "Checking service status..."
sudo systemctl status uniioai-proxy --no-pager -l | head -5

echo "✅ Update complete!"
