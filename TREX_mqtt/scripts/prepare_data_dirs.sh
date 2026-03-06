#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREX_MQTT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p \
  "$TREX_MQTT_DIR/data/postgres" \
  "$TREX_MQTT_DIR/data/pgadmin" \
  "$TREX_MQTT_DIR/postgres/databases"

if [[ $(id -u) -eq 0 ]]; then
  chown -R 5050:5050 "$TREX_MQTT_DIR/data/pgadmin"
  chmod 700 "$TREX_MQTT_DIR/data/pgadmin"
  echo "Prepared data directories and set pgAdmin ownership to 5050:5050"
else
  echo "Created directories. Run again as root if you need to set data/pgadmin ownership to 5050:5050."
fi
