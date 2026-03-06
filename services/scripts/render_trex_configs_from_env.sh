#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SERVICES_DIR/.." && pwd)"
ENV_FILE="${TREX_ENV_FILE:-$SERVICES_DIR/.env}"
RENDER_SCRIPT="$REPO_ROOT/TREX_Core/scripts/render_trex_db_config.py"
DEFAULT_CREDENTIALS_PATH="$REPO_ROOT/TREX_Core/configs/_credentials.json"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$RENDER_SCRIPT" ]]; then
  echo "TREX render script not found: $RENDER_SCRIPT" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.json> [<config.json> ...]" >&2
  exit 64
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

ARGS=(--credentials "$DEFAULT_CREDENTIALS_PATH")
for config_path in "$@"; do
  ARGS+=(--config "$config_path")
done

exec python3 "$RENDER_SCRIPT" "${ARGS[@]}"
