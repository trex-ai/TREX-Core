#!/bin/sh
set -eu

: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${PGADMIN_SERVER_JSON_FILE:=/tmp/servers.json}"

mkdir -p /var/lib/pgadmin "$(dirname "$PGADMIN_SERVER_JSON_FILE")"

python3 - <<'PY'
import json
import os
from pathlib import Path

host = os.environ.get("POSTGRES_HOST", "postgres")
user = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PASSWORD"]
server_json_file = Path(os.environ.get("PGADMIN_SERVER_JSON_FILE", "/var/lib/pgadmin/servers.json"))

pgpass_line = (
    host.replace("\\", "\\\\").replace(":", "\\:")
    + ":5432:*:"
    + user.replace("\\", "\\\\").replace(":", "\\:")
    + ":"
    + password.replace("\\", "\\\\").replace(":", "\\:")
)
Path("/tmp/pgpassfile").write_text(pgpass_line + "\n", encoding="utf-8")
os.chmod("/tmp/pgpassfile", 0o600)

servers = {
    "Servers": {
        "1": {
            "Name": "TREX PostgreSQL",
            "Group": "TREX",
            "Host": host,
            "Port": 5432,
            "MaintenanceDB": "postgres",
            "Username": user,
            "SSLMode": "prefer",
            "PassFile": "/tmp/pgpassfile",
        }
    }
}
server_json_file.parent.mkdir(parents=True, exist_ok=True)
server_json_file.write_text(json.dumps(servers, indent=2) + "\n", encoding="utf-8")
PY

exec /entrypoint.sh
