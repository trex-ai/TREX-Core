#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p \
  "$SERVICES_DIR/data/postgres" \
  "$SERVICES_DIR/postgres/databases"
echo "Prepared PostgreSQL data directories. pgAdmin state lives in the Docker volume pgadmin-data."
