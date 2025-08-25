#!/usr/bin/env bash
set -euo pipefail

# Activate per-user virtualenv if present, then exec the given command.
if [ -f "$HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$HOME/venv/bin/activate"
fi

exec "$@"
