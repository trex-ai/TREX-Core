# PostgreSQL Docker Requirements for TREX-Core

This document defines the requirements for adding PostgreSQL to the current TREX Docker stack.

It is intended to describe:

- the current Docker Compose setup in this repository
- the gap between that setup and the current host-installed PostgreSQL workflow
- the required shape of a Dockerized PostgreSQL + UI solution
- how database credentials and PostgreSQL settings should move into the existing `.env` workflow
- how database creation/restoration should become folder-driven

This is a requirements and design-direction document. It is not the final Compose implementation.

## 1. Current state

### 1.1 Current Compose stack in the repo

The only tracked Compose stack today is [`TREX_mqtt/docker-compose.yml`](TREX_mqtt/docker-compose.yml).

It currently defines these services:

- `mqtt` using EMQX
- `prometheus`
- `cadvisor`
- `mqtt-volume-metrics`

It also defines:

- one shared Docker network named `trex`
- bind-mounted persistence for EMQX and Prometheus data
- loopback-only host exposure by default via `127.0.0.1`
- a single `.env` file at [`TREX_mqtt/.env`](TREX_mqtt/.env)

Important current constraints:

- there is no PostgreSQL container in the stack
- there is no PostgreSQL UI container in the stack
- there are no PostgreSQL environment variables in [`TREX_mqtt/.env`](TREX_mqtt/.env)

### 1.2 Current PostgreSQL deployment

Today, PostgreSQL is installed directly on the host machine, not in Docker.

TREX currently expects to connect to PostgreSQL using config values like:

```json
"database": {
  "host": "localhost",
  "port": 5432,
  "connector": "postgresql+psycopg",
  "profiles_db": "citylearn_2022"
}
```

That is visible in [`TREX_Core/configs/citylearn_test.json`](TREX_Core/configs/citylearn_test.json).

The current codebase also reads credentials from:

- `TREX_Core/configs/_credentials.json`

That matters because moving PostgreSQL into Docker does not automatically move TREX off the `_credentials.json` path. If you want PostgreSQL credentials to live in `.env`, the containerized TREX startup path must either:

1. generate `_credentials.json` from environment variables, or
2. be refactored to read database credentials directly from environment variables

## 2. Source-of-truth assumptions from the codebase

These are design inputs derived from the current code and are mandatory.

### 2.1 TREX uses PostgreSQL for more than one database

TREX does not use a single static application database.

It uses:

- a read-only profile database such as `citylearn_2022`
- a per-study output database such as `citylearn_test3`
- optional replay databases from prior studies

### 2.2 The output database is created on demand

The runner creates the output database if it does not exist.

That means the PostgreSQL user used by TREX currently needs permission to create databases, unless the deployment changes the workflow to pre-create all output databases.

### 2.3 The profile database must already exist

TREX does not create or migrate `profiles_db`.

The Dockerized PostgreSQL design must support preloading profile databases before TREX starts.

### 2.4 Current code expects host/port/connector/profile DB in config

TREX currently expects:

- `database.host`
- `database.port`
- `database.connector`
- `database.profiles_db`

Those values are not currently loaded from environment variables by the app code.

### 2.5 Current code expects username/password separately

TREX currently expects:

- `username`
- `password`

in `_credentials.json`, not inside the main config JSON.

## 3. Goal state

The target Docker design must extend the existing stack so that:

1. PostgreSQL runs inside Docker instead of as host-installed software.
2. A PostgreSQL UI is available inside the same stack.
3. PostgreSQL credentials and PostgreSQL-related configuration live in [`TREX_mqtt/.env`](TREX_mqtt/.env).
4. New databases can be added by placing files in a folder, without editing Compose for each database.
5. The design remains compatible with the current `trex` Docker network and the repo's current loopback-first exposure style.

## 4. Version baseline

As of **March 6, 2026**:

- PostgreSQL's latest supported major version is **18**, and the current minor release is **18.3** according to PostgreSQL's release and versioning pages.
- pgAdmin 4's latest release is **9.12** according to pgAdmin's official release pages.

Requirements:

- the Dockerized database service should target **PostgreSQL 18**
- the UI should be **pgAdmin 4**
- committed Compose should prefer **explicit pinned tags** over a floating `latest` tag

Recommended examples:

- `postgres:18` or `postgres:18-bookworm`
- `dpage/pgadmin4:9.12`

Rationale:

- this satisfies the user's request for the latest PostgreSQL generation
- it avoids the drift and surprise rebuilds that come with committing `latest`

## 5. Required service additions

The Docker stack should be extended with at least these new services:

### 5.1 `postgres`

This service must:

- run PostgreSQL 18
- join the existing `trex` Docker network
- persist data across restarts
- expose PostgreSQL on port `5432`
- be reachable from other containers using a stable DNS name, preferably `postgres`
- include a healthcheck using `pg_isready`

### 5.2 `pgadmin`

This service must:

- run pgAdmin 4
- join the existing `trex` Docker network
- persist pgAdmin state across restarts
- be reachable from the host on a loopback-only bind by default
- be preconfigured to connect to the `postgres` service

### 5.3 `postgres-bootstrap` or equivalent

A separate bootstrap/import mechanism is required.

Reason:

- the official PostgreSQL Docker image supports initialization via environment variables and init scripts
- however, the official init-script mechanism only applies when the database directory is empty on first initialization
- that is not sufficient for the requirement "I should be able to add databases by just adding them to a folder"

Therefore the stack must include either:

- a one-shot sidecar service such as `postgres-bootstrap`, or
- a rerunnable bootstrap script invoked by Compose, or
- another equivalent idempotent database-loader process

This bootstrap/import mechanism must scan a mounted folder and create or restore databases from it.

## 6. Current Compose behavior that must be preserved

The PostgreSQL additions must preserve the current Compose posture:

- same stack file family under `TREX_mqtt/`
- same `.env` file driven configuration style
- same `trex` network
- same loopback-only host binding by default
- same persistence style using bind mounts or clearly named volumes
- same `restart: unless-stopped` operational model

## 7. Environment variable requirements

All PostgreSQL-related configuration should move into [`TREX_mqtt/.env`](TREX_mqtt/.env).

That includes:

### 7.1 PostgreSQL image/runtime settings

- `POSTGRES_IMAGE_TAG`
- `POSTGRES_BIND_ADDRESS`
- `POSTGRES_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `POSTGRES_INITDB_ARGS`
- `POSTGRES_HOST`
- `POSTGRES_CONTAINER_NAME`
- `POSTGRES_CPUS`
- `POSTGRES_MEM_LIMIT`

### 7.2 pgAdmin settings

- `PGADMIN_IMAGE_TAG`
- `PGADMIN_BIND_ADDRESS`
- `PGADMIN_PORT`
- `PGADMIN_DEFAULT_EMAIL`
- `PGADMIN_DEFAULT_PASSWORD`
- `PGADMIN_LISTEN_PORT`
- `PGADMIN_CPUS`
- `PGADMIN_MEM_LIMIT`

### 7.3 TREX-facing database config

Because TREX currently stores DB host/port/connector/profile DB settings in JSON config files, the deployment must also define env keys that represent those values, for example:

- `TREX_DB_HOST`
- `TREX_DB_PORT`
- `TREX_DB_CONNECTOR`
- `TREX_PROFILES_DB`
- `TREX_DB_USERNAME`
- `TREX_DB_PASSWORD`

Requirement:

- the Dockerized TREX startup path must consume these values and map them into the format TREX expects

Acceptable ways to do that:

1. Generate `TREX_Core/configs/_credentials.json` from env at startup and template the JSON config database block.
2. Refactor TREX to read DB settings directly from env.

The first option is the lowest-risk path because it avoids broad code changes.

## 8. Networking requirements

### 8.1 Host access

By default, PostgreSQL and pgAdmin should be bound to `127.0.0.1`, not to all interfaces.

That matches the current MQTT stack posture.

### 8.2 Container-to-container access

Inside Docker, PostgreSQL should be reachable as:

```text
postgres:5432
```

pgAdmin should connect to PostgreSQL using the service DNS name, not host networking.

### 8.3 Host-run TREX during transition

During the migration phase, TREX may still be running on the host while PostgreSQL moves into Docker.

In that transition state:

- host-run TREX can still connect through the published loopback address, e.g. `127.0.0.1:5432`

After TREX itself is containerized:

- TREX configs should switch to `postgres:5432`

This distinction must be documented clearly in the implementation.

## 9. Persistence requirements

The design must persist:

### 9.1 PostgreSQL data

Persist PostgreSQL data under a stable path, for example:

- `TREX_mqtt/data/postgres`

### 9.2 pgAdmin state

Persist pgAdmin state under a stable path, for example:

- `TREX_mqtt/data/pgadmin`

### 9.3 Database input folder

Database definitions and restores should live under a mounted repo path, for example:

- `TREX_mqtt/postgres/databases`

This folder is source input, not live database storage.

## 10. Folder-driven database provisioning requirements

This is the most important functional requirement in this document.

You should be able to add a database by creating a folder under a designated directory, without editing Compose for each new database.

### 10.1 Proposed folder convention

Recommended structure:

```text
TREX_mqtt/postgres/
  databases/
    citylearn_2022/
      001-schema.sql
      010-data.sql
    citylearn_test3/
      dump.sql
    some_other_db/
      restore.dump
```

Folder name requirement:

- each direct child directory name becomes the target database name

Allowed contents:

- `.sql`
- `.sql.gz`
- `.dump`
- `.tar`
- optionally a metadata file if the bootstrap process needs one

### 10.2 Bootstrap behavior

The bootstrap/import mechanism must:

1. wait for PostgreSQL to become healthy
2. scan the configured database folder
3. create any missing database whose folder name is not already present
4. load that database's SQL or dump files
5. avoid re-importing databases that were already successfully loaded
6. be safe to rerun after every `docker compose up`

### 10.3 Idempotency requirement

This process must be idempotent.

It must not:

- destroy existing databases just because the loader reruns
- blindly replay every `.sql` file on every restart
- rely only on `/docker-entrypoint-initdb.d`

Recommended acceptable implementations:

- track imported databases in a control table
- or track imported folders/files with a manifest file inside PostgreSQL
- or only bootstrap if the target database does not exist

### 10.4 Operational model

Adding a new database should look like this:

1. create a new folder under `TREX_mqtt/postgres/databases/`
2. place SQL or dump files inside it
3. rerun the bootstrap service, or rerun `docker compose up -d`
4. the new database appears in PostgreSQL and pgAdmin

This workflow must not require:

- editing the Compose file
- adding a new service
- manually running `createdb` by hand
- manually copying files into the container

## 11. UI requirements

The PostgreSQL UI should be pgAdmin 4, not a generic DB browser.

Reasons:

- it is purpose-built for PostgreSQL
- it supports saved server connections
- it supports backup and restore workflows
- it fits the user's requirement for "Postgres with UI"

pgAdmin must be configured so that:

- the initial admin account is created from env values
- its data persists across restarts
- the default server entry points at the `postgres` container
- it is reachable from the host on loopback only by default

## 12. TREX integration requirements

### 12.1 Database hostname handling

The design must account for the fact that `localhost` means different things depending on where TREX runs.

If TREX runs on the host:

- `localhost:5432` can still work against a published PostgreSQL container port

If TREX runs in Docker:

- `localhost` is wrong
- `postgres` must be used as the DB host

### 12.2 Credentials handling

Since the user wants PostgreSQL username/password in `.env`, the final solution must stop relying on manually edited secret files as the primary source of truth.

Minimum acceptable outcome:

- `.env` is the source of truth
- TREX startup translates those env values into whatever the current code expects

### 12.3 Profile DB support

The folder-driven database loader must support importing profile databases like:

- `citylearn_2022`

without special-casing them in Compose.

## 13. Migration from host-installed PostgreSQL

The design should support migration from the current host-installed PostgreSQL instance into the Dockerized one.

Recommended migration shape:

1. export databases from the host PostgreSQL installation
2. place each exported database into its own folder under the configured database-import directory
3. bring up the Dockerized PostgreSQL stack
4. let the bootstrap service create and restore the databases
5. point TREX at the Dockerized PostgreSQL endpoint

This lets the repo become the operational home for database imports instead of the host package manager installation.

## 14. Recommended folder layout in the repo

The implementation should use a structure similar to:

```text
TREX_mqtt/
  docker-compose.yml
  .env
  data/
    emqx/
    prometheus/
    postgres/
    pgadmin/
  postgres/
    databases/
      citylearn_2022/
      citylearn_test3/
    bootstrap/
      apply_databases.sh
      servers.json
```

Notes:

- `data/postgres` is the live PostgreSQL data directory
- `data/pgadmin` is the live pgAdmin state directory
- `postgres/databases` is the declarative import source
- `postgres/bootstrap` contains helper scripts and optional pgAdmin server definitions

## 15. Acceptance criteria

The design is acceptable only if all of the following are true.

### 15.1 Stack behavior

- `docker compose up -d` brings up PostgreSQL and pgAdmin in the existing stack
- PostgreSQL becomes healthy before the bootstrap/import process runs
- pgAdmin can connect to PostgreSQL without host networking hacks

### 15.2 Configuration behavior

- PostgreSQL credentials are defined in [`TREX_mqtt/.env`](TREX_mqtt/.env)
- PostgreSQL host/port/image/UI settings are defined in [`TREX_mqtt/.env`](TREX_mqtt/.env)
- no per-database Compose edits are required

### 15.3 Database provisioning behavior

- placing a new folder under the database-import directory is enough to declare a new database
- the bootstrap/import mechanism creates or restores that database
- rerunning the stack does not destroy already-loaded databases

### 15.4 TREX compatibility

- host-run TREX can connect during transition
- container-run TREX can connect via service DNS later
- profile databases remain available
- output study databases can still be created as required by TREX, unless the design intentionally pre-creates them and documents that change

## 16. Implementation recommendations

The eventual implementation should prefer:

- `postgres:18` or `postgres:18-bookworm`
- `dpage/pgadmin4:9.12`
- a `postgres` service name
- a `pgadmin` service name
- a rerunnable `postgres-bootstrap` helper service
- env-driven configuration
- loopback-only host exposure
- persisted bind mounts under `TREX_mqtt/data/`

It should avoid:

- floating `latest` tags committed to the repo
- storing PostgreSQL credentials only in `_credentials.json`
- relying only on `/docker-entrypoint-initdb.d` for folder-driven database additions
- exposing PostgreSQL or pgAdmin publicly by default

## 17. Source references

These current-version notes were verified on **March 6, 2026** from official sources:

- PostgreSQL versioning and current releases: <https://www.postgresql.org/support/versioning/>
- PostgreSQL 18.3 release notes: <https://www.postgresql.org/docs/release/18.3/>
- PostgreSQL release announcement on February 26, 2026: <https://www.postgresql.org/about/news/postgresql-183-179-1613-1517-and-1422-released-3246/>
- Official PostgreSQL Docker image environment variable behavior: <https://hub.docker.com/_/postgres/>
- pgAdmin container deployment docs: <https://www.pgadmin.org/docs/pgadmin4/latest/container_deployment.html>
- pgAdmin 4 release archive showing v9.12: <https://www.pgadmin.org/news/>
