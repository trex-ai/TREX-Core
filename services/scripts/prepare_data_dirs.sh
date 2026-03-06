#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p \
  "$SERVICES_DIR/data/postgres" \
  "$SERVICES_DIR/data/pgadmin" \
  "$SERVICES_DIR/postgres/databases"

if [[ $(id -u) -eq 0 ]]; then
  chown -R 5050:5050 "$SERVICES_DIR/data/pgadmin"
  chmod 700 "$SERVICES_DIR/data/pgadmin"
  echo "Prepared data directories and set pgAdmin ownership to 5050:5050"
else
  echo "Created directories. Run again as root if you need to set data/pgadmin ownership to 5050:5050."
fi
