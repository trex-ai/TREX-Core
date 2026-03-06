# TREX PostgreSQL + pgAdmin implementation

This tree provides the PostgreSQL, pgAdmin, and folder-driven database bootstrap implementation requested by the requirements document.

Because the original repository was **not mounted** in this environment, the Compose work is delivered as a drop-in overlay file:

- `docker-compose.postgres.yml`

Use it either:

- alongside your existing `TREX_mqtt/docker-compose.yml` (preferred), or
- by itself if you only want to stand up the PostgreSQL/pgAdmin portion first.

If you switch between the standalone and combined Compose invocations, stop the earlier stack first so Docker does not hit fixed `container_name` conflicts.

## What is included

- `docker-compose.postgres.yml`
  - `postgres` service on the existing `trex` network
  - `pgadmin` service, loopback-only by default
  - `postgres-bootstrap` controller that scans `postgres/databases/`
  - Docker-managed `pgadmin-data` volume for pgAdmin state
- `.env` and `.env.example`
  - all PostgreSQL and pgAdmin settings
  - TREX-facing database settings
- `postgres/bootstrap/apply_databases.sh`
  - idempotent folder scanner/importer
- `postgres/bootstrap/render_pgadmin_bootstrap.sh`
  - generates pgAdmin `servers.json` and `.pgpass` from env at startup
- `scripts/render_trex_configs_from_env.sh`
  - sources `TREX_mqtt/.env` and renders TREX JSON/credential files
- `../TREX_Core/scripts/render_trex_db_config.py`
  - updates `database.host`, `database.port`, `database.connector`, and `database.profiles_db`
  - writes `TREX_Core/configs/_credentials.json`
- `../TREX_Core/scripts/entrypoint_with_db_env.sh`
  - future container-entrypoint wrapper for TREX

## Quick start

1. Review `TREX_mqtt/.env` and replace the development credentials/passwords.
2. Keep `TREX_DB_USERNAME` and `TREX_DB_PASSWORD` aligned with `POSTGRES_USER` and `POSTGRES_PASSWORD`.
3. Place any preloaded profile databases under `TREX_mqtt/postgres/databases/<db-name>/`.
4. Start the database stack. Preferred, alongside the existing MQTT stack:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
   ```

   Or, if you want to start only the PostgreSQL portion first:

   ```bash
   docker compose -f docker-compose.postgres.yml up -d
   ```

5. Open pgAdmin on `http://127.0.0.1:${PGADMIN_PORT}` and sign in with `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` from `TREX_mqtt/.env`.

pgAdmin is preloaded with a server definition for `postgres:5432`, so no manual server registration is required after login.

## Folder-driven database provisioning

The bootstrap controller runs continuously and rescans `postgres/databases/` every `POSTGRES_BOOTSTRAP_INTERVAL_SECONDS` seconds.

To add a new database:

1. Create a new folder under `TREX_mqtt/postgres/databases/`.
2. Put one of these inside it:
   - a single full `pg_dump` SQL file (`.sql` or `.sql.gz`) that includes `CREATE DATABASE`, or
   - database-local `.sql`, `.sql.gz`, `.dump`, or `.tar` files to be applied into a newly created database.
3. Wait for the next scan, or force a manual scan with:

   ```bash
   docker exec trex-postgres-bootstrap /bin/bash /bootstrap/apply_databases.sh --once
   ```

Existing databases are never dropped or blindly replayed by the controller.

## TREX integration

### Host-run TREX during transition

Keep these values in `TREX_mqtt/.env`:

```dotenv
TREX_DB_HOST=127.0.0.1
TREX_DB_PORT=5432
```

Also keep:

```dotenv
TREX_DB_USERNAME=${POSTGRES_USER}
TREX_DB_PASSWORD=${POSTGRES_PASSWORD}
```

Then render TREX's config and credentials files from the shared `.env` source of truth:

```bash
./scripts/render_trex_configs_from_env.sh ../TREX_Core/configs/citylearn_test.json
```

That updates:

- the `database` block inside the JSON config file
- `TREX_Core/configs/_credentials.json`

### Future containerized TREX

When TREX later moves into Docker, switch the env file to:

```dotenv
TREX_DB_HOST=postgres
TREX_DB_PORT=5432
```

Then use `TREX_Core/scripts/entrypoint_with_db_env.sh` as the TREX container entrypoint wrapper so the same env values continue to generate the JSON files TREX already expects.

## Idempotency model

The bootstrap controller uses two checks:

- whether the target database already exists
- whether the controller has a successful import record in `trex_bootstrap.imported_databases`

The controller imports a folder only when the target database is missing.

If a database exists without a successful controller record, it is treated as external/manual and left untouched.

If a failed import leaves the target database behind, drop that database before retrying so the controller can recreate and re-import it on the next scan.

## Filesystem layout

```text
TREX_mqtt/
  docker-compose.postgres.yml
  .env
  data/
    postgres/
  postgres/
    databases/
    bootstrap/
      apply_databases.sh
      render_pgadmin_bootstrap.sh
  scripts/
    render_trex_configs_from_env.sh
    run_postgres_bootstrap_once.sh
```

pgAdmin data is stored in the Docker volume `pgadmin-data`, not in a repository bind mount.

## Notes

- `POSTGRES_PORT` is the **host-published** port. Container-to-container access remains `postgres:5432`.
- pgAdmin is preloaded with a server definition that points to `postgres:5432` and uses a generated `.pgpass` file.
- pgAdmin now uses the Docker volume `pgadmin-data`; the old `data/pgadmin/` bind mount is no longer part of the stack.
- Full `pg_dump` SQL dumps that include `CREATE DATABASE` are supported when they are the only file in a database folder.
- For large restores, custom-format (`.dump`) or tar-format (`.tar`) archives are still preferred.
- If you change the contents of an already imported database folder and want to reload it, drop the target database; the controller will recreate it on the next scan.
