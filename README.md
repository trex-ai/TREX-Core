# TREX-Core
**TREX is a simulation framework for developing Transactive Energy systems**

## Python Environment with `uv`

Install the project dependencies into a local virtual environment from the repository root:

```bash
uv python install 3.11
uv sync
source .venv/bin/activate
```

## Docker Services for MQTT and PostgreSQL

Start the local service stack from [`services/`](/Users/mikael/Documents/work-projects/TREX-Core/services):

```bash
cd services
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
```

This brings up the MQTT broker, PostgreSQL, pgAdmin, and the supporting monitoring/bootstrap services.

## Import PostgreSQL Databases

Database imports are folder-driven. The folder name becomes the database name, and supported dump formats are `.sql`, `.sql.gz`, `.dump`, and `.tar`.

```bash
mkdir -p services/postgres/databases/<db-name>
cp /path/to/dump.sql services/postgres/databases/<db-name>/
cd services
./scripts/run_postgres_bootstrap_once.sh
```

If the stack is already running, the bootstrap service can also pick up new database folders automatically on its next scan.
