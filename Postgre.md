# PostgreSQL Notes For TREX-Core

This document captures how PostgreSQL is used in the current codebase and what needs to be true before moving the database behind a Docker container.

## 1. Current architecture

TREX-Core uses PostgreSQL for three distinct purposes:

1. A profile database that already exists and is treated as read-only at runtime.
2. A per-study output database that TREX creates on demand.
3. An optional replay source database, which is just another study output database from an earlier run.

The current default entrypoint is [`TREX_Core/main.py`](TREX_Core/main.py), which runs:

- config: `citylearn_test`
- purge: `True`
- simulations: `baseline`

That means the default run path currently expects:

- a PostgreSQL server reachable from `TREX_Core/configs/citylearn_test.json`
- an existing profile database named `citylearn_2022`
- permission to drop and recreate the output database named `citylearn_test3`

## 2. Libraries and connection stack

PostgreSQL access is split across sync and async libraries:

- `sqlalchemy`
- `sqlalchemy-utils`
- `psycopg[binary]`
- `databases[postgresql]`
- `asyncpg`

See [`pyproject.toml`](pyproject.toml).

The code builds connection strings in [`TREX_Core/utils/db_utils.py`](TREX_Core/utils/db_utils.py) with this shape:

```text
postgresql+psycopg://<username>:<password>@<host>:<port>/<db_name>
```

Important implications:

- `database.port` is effectively required at runtime.
- Username and password are interpolated directly into the URL.
- If the password contains URL-reserved characters like `@`, `:`, or `/`, the current code does not escape them.

## 3. Config and credential discovery

### Config JSON discovery

Config lookup is handled by:

- [`TREX_Core/runner/runner.py`](TREX_Core/runner/runner.py)
- [`TREX_Core/utils/db_utils.py`](TREX_Core/utils/db_utils.py)

Lookup order:

1. `TREX_CORE_ROOT`
2. current working directory
3. current working directory + `/TREX_Core`
4. the package root

This matters in Docker because app containers need either:

- the repo mounted at a predictable path, or
- `TREX_CORE_ROOT` set so the config files can be found reliably

### Credentials

Database credentials are read from:

- `TREX_Core/configs/_credentials.json`

Only the example file is present in the repo:

- [`TREX_Core/configs/_credentials.json.example`](TREX_Core/configs/_credentials.json.example)

Expected shape:

```json
{
  "username": "your_user",
  "password": "your_password"
}
```

There is no environment-variable based database credential loader in the current code.

## 4. Config inventory

### DB-ready config files

The tracked configs that define a `database` block are:

- [`TREX_Core/configs/_template.json`](TREX_Core/configs/_template.json)
- [`TREX_Core/configs/citylearn_test.json`](TREX_Core/configs/citylearn_test.json)
- [`TREX_Core/configs/citylearn_test3.json`](TREX_Core/configs/citylearn_test3.json)

Current values:

| File | Host | Port | Connector | Profiles DB | Notes |
| --- | --- | --- | --- | --- | --- |
| `TREX_Core/configs/_template.json` | `localhost` | commented out | `postgresql+psycopg` | `profiles` | Template comment is stale; port must actually be set |
| `TREX_Core/configs/citylearn_test.json` | `localhost` | `5432` | `postgresql+psycopg` | `citylearn_2022` | Used by `TREX_Core/main.py` |
| `TREX_Core/configs/citylearn_test3.json` | `localhost` | `5432` | `postgresql+psycopg` | `citylearn_2022` | Same DB pattern, longer run |

### Config that is not DB-ready

[`TREX_Core/configs/mqtt_test.json`](TREX_Core/configs/mqtt_test.json):

- has no `database` block
- is version `5.0.0`
- does not match the current runner expectation of config version `>= 5.1.0`

## 5. Runtime database model

### 5.1 Profile database

Configured by:

- `config["database"]["profiles_db"]`

Used by:

- [`TREX_Core/runner/runner.py`](TREX_Core/runner/runner.py) to validate interval consistency before launching
- [`TREX_Core/participants/base.py`](TREX_Core/participants/base.py) to read load/generation profiles during the simulation

The profile DB is not created by TREX. It must already exist.

The code expects one table per participant profile:

- table name = participant ID, or
- table name = `trader.use_synthetic_profile`

Required profile table column assumptions:

- `time`
- either:
  - `generation` and `consumption`, or
  - `grid` and `solar+`

The profile read path queries:

- `WHERE time = <interval_end_timestamp>`

### 5.2 Output study database

The runner injects:

```json
"database": {
  "output_db": "<study_name>"
}
```

This is done at runtime from `study.name`. It is not normally written in the source config file.

The output DB is created on demand by [`TREX_Core/runner/runner.py`](TREX_Core/runner/runner.py) via [`TREX_Core/utils/db_utils.py`](TREX_Core/utils/db_utils.py).

One output database is created per study name.

Example with `citylearn_test.json`:

- `study.name = "citylearn_test3"`
- output DB name becomes `citylearn_test3`

### 5.3 Replay database

If a participant trader is configured with `actions.replay`, participants open another study database and read:

- table name: `<episode>_<market>_records`
- columns used: `time`, `participant_id`, `next_actions`

This replay DB must already exist before the replay run starts.

## 6. What each process does to PostgreSQL

### Runner

[`TREX_Core/runner/runner.py`](TREX_Core/runner/runner.py):

- resolves the config file
- injects `database.output_db = study.name`
- builds the profile DB connection string
- samples up to 5 profile tables to infer the simulation interval
- builds the output DB connection string
- optionally drops the output DB when `purge=True`
- creates the output DB if missing
- creates the `configs` table if missing
- inserts the original config JSON into `configs`

Important: the `configs` table stores `self.config_original`, not the runtime-mutated config. That means it does not capture injected values like:

- `database.output_db`
- derived `study.time_step_size`
- derived `study.start_time`
- derived `study.total_steps`
- simulation-specific changes made in `modify_config()`

### Sim controller

[`TREX_Core/sim_controller/sim_controller.py`](TREX_Core/sim_controller/sim_controller.py):

- if `records` is enabled, it instantiates `Records`
- at the start of each episode it creates the shared participant records table

It does not actually write rows to that table itself. Participants do.

### Participants

[`TREX_Core/participants/base.py`](TREX_Core/participants/base.py) and [`TREX_Core/participants/client.py`](TREX_Core/participants/client.py):

- open the profile DB on connect
- optionally open a replay records DB on connect
- open the output records table on episode start
- buffer participant records and flush them asynchronously
- flush records at episode end
- close connections at simulation end

### Market

[`TREX_Core/markets/base/DoubleAuction.py`](TREX_Core/markets/base/DoubleAuction.py) and [`TREX_Core/markets/client.py`](TREX_Core/markets/client.py):

- open the output DB on episode start
- create the per-episode market transaction table if missing
- buffer transaction inserts asynchronously
- flush at episode end
- verify row count after the flush
- close the DB connection at simulation end

## 7. Table lifecycle and naming

### 7.1 Always-created output table

The runner creates this once per output DB:

#### `configs`

Columns:

- `id INTEGER PRIMARY KEY`
- `data JSON`

Inserted row:

- `id = 0`
- `data = original config JSON`

### 7.2 Legacy or inactive metadata path

There is helper code for a `metadata` table, but it is not part of the normal run path today.

Problems with the current metadata path:

- `__create_sim_metadata()` is not called by the runner
- some code still references `study.output_database`, which is no longer the live configuration path
- `update_metadata()` expects `md_table.c.generation`, but the defined table schema uses `episode`

Treat `metadata` as legacy/inactive unless you plan to repair that path.

### 7.3 Per-episode market transaction table

Created by the market process:

```text
<episode>_<market_id>_market
```

Example:

```text
1_baseline_market
```

Schema from [`TREX_Core/utils/db_utils.py`](TREX_Core/utils/db_utils.py):

- `id INTEGER PRIMARY KEY`
- `quantity INTEGER`
- `seller_id STRING`
- `buyer_id STRING`
- `energy_source STRING`
- `settlement_price_sell FLOAT`
- `settlement_price_buy FLOAT`
- `fee_ask FLOAT`
- `fee_bid FLOAT`
- `time_creation INTEGER`
- `time_purchase INTEGER`
- `time_consumption INTEGER`

Notes:

- `id` is auto-generated by PostgreSQL because inserts do not supply it explicitly.
- `fee_ask` and `fee_bid` exist in the schema but do not appear to be populated by the current market code.

### 7.4 Per-episode participant records table

Created by the sim controller and written by participants:

```text
<episode>_<market_id>_records
```

Example:

```text
1_baseline_records
```

This is a shared table for all participants in the episode. It is not one table per participant.

Base columns always present:

- `time INTEGER PRIMARY KEY`
- `participant_id STRING PRIMARY KEY`

Optional columns depend on `config["records"]`:

- `meter JSON`
- `next_actions JSON`
- `metadata JSON`
- `storage_info JSON`
- `remaining_energy INTEGER`
- `state_of_charge FLOAT`

Notes:

- The primary key is composite: `(time, participant_id)`.
- `remaining_energy` and `state_of_charge` depend on `storage_info`.

### 7.5 Profile tables

Profile tables are not created by TREX.

The runner samples them to infer timestep size, and participants read them each round.

Expected naming:

- participant ID, or
- `trader.use_synthetic_profile`

Expected columns:

- `time`
- and either:
  - `generation`, `consumption`
  - or `grid`, `solar+`

## 8. Buffering and write behavior

### Market transaction writes

Market writes are buffered in memory and flushed when:

- at least `10000` transactions are buffered, or
- episode end forces a final flush

### Participant record writes

Participant record writes are buffered in memory and flushed when:

- at least `1000` records are buffered, or
- episode end forces a final flush

This means container shutdowns and restarts need graceful termination. Hard kills can lose buffered rows that have not been flushed yet.

## 9. Docker-specific facts from the repo today

There is currently no PostgreSQL container in the repo.

The only Compose stack is [`services/docker-compose.yml`](services/docker-compose.yml), which defines:

- `mqtt`
- `prometheus`
- `cadvisor`
- `mqtt-volume-metrics`

It already provides a shared Docker network named:

- `trex`

The existing MQTT env file [`services/.env`](services/.env) contains no PostgreSQL settings.

## 10. Migration risks and constraints

### 10.1 Startup ordering matters

There is no robust retry/backoff layer around PostgreSQL startup. If TREX connects before Postgres is ready, startup can fail.

For Docker you should have at least:

- a PostgreSQL healthcheck
- app startup that waits for PostgreSQL readiness
- `depends_on` or an explicit wait script

### 10.2 The app user may need `CREATEDB`

TREX creates the output study database automatically.

That means the DB user in `_credentials.json` needs permission to create databases, unless you change the workflow to pre-create every output DB yourself.

### 10.3 The profile DB must be preloaded

TREX never creates `profiles_db`. It must already contain:

- the named database
- the expected profile tables
- enough rows at and after the configured `start_datetime`

### 10.4 `database.port` must be explicit

The live code uses `db_config['port']` directly.

The template file currently comments out the port and even shows `1883`, which is the MQTT port, not PostgreSQL. For Dockerized Postgres, set this explicitly to `5432` unless you intentionally expose a different container port.

### 10.5 Config files must still be accessible inside the container

Database credentials are not pulled from environment variables. TREX reads:

- config JSON files from `TREX_Core/configs`
- credentials from `TREX_Core/configs/_credentials.json`

That means you must either:

- mount those files into the container, or
- bake them into the image, or
- change the code

### 10.6 `database.output_db` is injected, not authored

If you run participant or market processes directly and bypass the runner, they still expect `database_config["output_db"]` to exist.

That field is injected by the runner, not stored in the tracked JSON files.

### 10.7 Legacy `study.output_database` references are stale

Some old code still references `config["study"]["output_database"]`, but the active flow now uses:

- `config["database"]["output_db"]`
- `db_utils.make_db_str(...)`

Do not build a Docker migration around `study.output_database`.

### 10.8 Parallel runs can create awkward table names

When `study.parallel > 1`, the runner appends `/<idx>` to `market.id`.

That value flows into table names:

```text
<episode>_<market_id>_market
<episode>_<market_id>_records
```

So a parallel run can produce table names containing `/`.

PostgreSQL can store quoted identifiers like that, but it is awkward for:

- ad hoc SQL
- backup filters
- restore scripts
- shell tooling

### 10.9 Buffered writes mean graceful shutdown matters

Because records and transactions are buffered in memory, container orchestration should allow time for:

- episode-end flushes
- `ensure_records_complete()`
- `ensure_transactions_complete()`

Abrupt container termination can lose buffered data.

## 11. Estimated connection footprint

Per participant process:

- 1 connection to the profile DB
- plus 1 connection to the output records DB when `records` is enabled
- plus 1 replay DB connection when action replay is enabled

Shared processes:

- market: 1 output DB connection
- sim controller: no long-lived DB write connection in the current code path
- runner: short-lived sync SQLAlchemy connections for setup and profile checks

A rough rule of thumb for max connections is:

```text
participants * (1 + records_enabled + replay_enabled) + 1 market + setup overhead
```

Leave headroom for SQLAlchemy inspection calls and admin access.

## 12. What to change for a Dockerized PostgreSQL deployment

At minimum:

1. Add a PostgreSQL service to Docker on the same network as the TREX app containers.
2. Change each TREX config file's `database.host` from `localhost` to the Postgres service DNS name, for example `postgres`.
3. Keep `database.port` explicit, normally `5432`.
4. Keep `server.host` as the MQTT service DNS name inside Docker, typically `mqtt`.
5. Provide `TREX_Core/configs/_credentials.json` inside the app container.
6. Preload the profile database into PostgreSQL before launching TREX.
7. Decide whether the TREX DB user is allowed to create output databases.
8. Add Postgres readiness checks before TREX processes start.
9. Persist Postgres data on a Docker volume.
10. Preserve graceful shutdown so buffered writes flush cleanly.

## 13. Pre-migration checklist

- [ ] A real `TREX_Core/configs/_credentials.json` exists for the containerized app.
- [ ] The Postgres container is reachable by DNS name from TREX containers.
- [ ] `database.host` uses the Postgres service name, not `localhost`.
- [ ] `database.port` is set to `5432` or your real container port.
- [ ] The profile database exists inside Postgres.
- [ ] The profile tables exist for all participant IDs or synthetic profile names.
- [ ] The profile tables contain a `time` column and either `generation/consumption` or `grid/solar+`.
- [ ] The DB user can create the output DB, or the output DB is pre-created.
- [ ] TREX app containers can read the config directory.
- [ ] Startup waits for Postgres readiness.
- [ ] Shutdown allows enough time for buffered inserts to flush.

## 14. Short summary

The key point is that TREX does not use PostgreSQL as a single static schema-managed database. It uses:

- one existing read-only profile DB
- one lazily created output DB per study
- dynamically created per-episode tables inside that output DB

So the Docker migration is not just "move one database to a container". You need to preserve:

- config discovery
- credential file discovery
- profile DB preload
- output DB creation permissions
- service DNS names
- startup ordering
- graceful shutdown for buffered writes
