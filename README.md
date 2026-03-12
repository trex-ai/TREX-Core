# TREX-Core
**TREX is a simulation framework for developing Transactive Energy systems**

## Python Environment with `uv`

Install the project dependencies into a local virtual environment from the repository root:

```bash
uv python install 3.12
uv sync
source .venv/bin/activate
```

## Install pre-commit hooks
```bash
uv run pre-commit install
```
to run the pre-commit hooks on all files:
```bash
uv run pre-commit run --all-files
```
to run the pre-commit on specific files:
```bash
uv run pre-commit run --files path/to/your_file.py
```
If you want to run only a specific hook on that file:
```bash
uv run pre-commit run ruff-check --files path/to/your_file.py
uv run pre-commit run mypy --files path/to/your_file.py
```
## Useful Ruff commands
To check for linting issues:
```bash
uv run ruff check .
```
To automatically fix issues:
```bash
uv run ruff check . --fix
```
to get the stats of the issues:
```bash
uv run ruff check . --statistics
```

## Handle detect-secrets baseline if it doesn't exist yet
```bash
uv run detect-secrets scan > .secrets.baseline
```


## Run the test suite
```bash
uv run pytest
```
Since your pytest config has --cov-fail-under=7 and markers for integration tests that need MQTT/PostgreSQL, you can skip those initially:
```bash
uv run pytest -m "not integration"
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
