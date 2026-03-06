#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREX_MQTT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TREX_MQTT_DIR"

BOOTSTRAP_CONTAINER="trex-postgres-bootstrap"
BOOTSTRAP_CMD=(/bin/bash /bootstrap/apply_databases.sh --once)

if ! docker inspect "$BOOTSTRAP_CONTAINER" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
PostgreSQL bootstrap container is not present.
Start the stack first with:
  docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
EOF
  exit 1
fi

BOOTSTRAP_STATUS="$(docker inspect -f '{{.State.Status}}' "$BOOTSTRAP_CONTAINER")"
if [[ "$BOOTSTRAP_STATUS" != "running" ]]; then
  printf 'PostgreSQL bootstrap container is %s, not running.\n' "$BOOTSTRAP_STATUS" >&2
  echo "Start or recreate the stack first with:" >&2
  echo "  docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d" >&2
  exit 1
fi

exec docker exec "$BOOTSTRAP_CONTAINER" "${BOOTSTRAP_CMD[@]}"
