#!/bin/sh
set -eu

: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${PGADMIN_SERVER_JSON_FILE:=/tmp/servers.json}"

mkdir -p /var/lib/pgadmin "$(dirname "$PGADMIN_SERVER_JSON_FILE")"

# pgAdmin's --replace import creates a fresh server row on every startup,
# which changes the internal server id and breaks any saved browser tabs.
if [ -f /var/lib/pgadmin/pgadmin4.db ]; then
  export PGADMIN_REPLACE_SERVERS_ON_STARTUP="False"
else
  export PGADMIN_REPLACE_SERVERS_ON_STARTUP="True"
fi

python3 - <<'PY'
import json
import os
from pathlib import Path

host = os.environ.get("POSTGRES_HOST", "postgres")
user = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PASSWORD"]
pgadmin_email = os.environ["PGADMIN_DEFAULT_EMAIL"]
server_json_file = Path(os.environ.get("PGADMIN_SERVER_JSON_FILE", "/var/lib/pgadmin/servers.json"))


def preprocess_username(username: str) -> str:
    if len(username) == 0 or username[0].isdigit():
        username = "pga_user_" + username
    return username.replace("@", "_").replace("/", "slash").replace("\\", "slash")


pgadmin_storage_dir = Path("/var/lib/pgadmin/storage") / preprocess_username(pgadmin_email)
pgadmin_storage_dir.mkdir(parents=True, exist_ok=True)
pgpass_file = pgadmin_storage_dir / "pgpass"
imported_pgpass_path = "/pgpass"

pgpass_line = (
    host.replace("\\", "\\\\").replace(":", "\\:")
    + ":5432:*:"
    + user.replace("\\", "\\\\").replace(":", "\\:")
    + ":"
    + password.replace("\\", "\\\\").replace(":", "\\:")
)
pgpass_file.write_text(pgpass_line + "\n", encoding="utf-8")
os.chmod(pgpass_file, 0o600)

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
            "PassFile": imported_pgpass_path,
        }
    }
}
server_json_file.parent.mkdir(parents=True, exist_ok=True)
server_json_file.write_text(json.dumps(servers, indent=2) + "\n", encoding="utf-8")
PY

exec /entrypoint.sh
