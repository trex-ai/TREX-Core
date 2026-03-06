#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="$SCRIPT_DIR/render_trex_db_config.py"

CONFIG_PATHS_RAW="${TREX_CONFIG_PATHS:-${TREX_CONFIG_PATH:-}}"
if [[ -z "$CONFIG_PATHS_RAW" ]]; then
  echo "TREX_CONFIG_PATH or TREX_CONFIG_PATHS must be set" >&2
  exit 64
fi

IFS=':' read -r -a CONFIG_PATHS <<< "$CONFIG_PATHS_RAW"
CREDENTIALS_PATH="${TREX_CREDENTIALS_PATH:-$(dirname "${CONFIG_PATHS[0]}")/_credentials.json}"

ARGS=(--credentials "$CREDENTIALS_PATH")
for config_path in "${CONFIG_PATHS[@]}"; do
  ARGS+=(--config "$config_path")
done

python3 "$RENDER_SCRIPT" "${ARGS[@]}"
exec "$@"
