#!/usr/bin/env bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$DIR/.venv/bin/activate" ]; then
  . "$DIR/.venv/bin/activate"
elif [ -f "$DIR/.venv/Scripts/activate" ]; then
  . "$DIR/.venv/Scripts/activate"
else
  echo "Activation script not found in $DIR/.venv"
  exit 1
fi
