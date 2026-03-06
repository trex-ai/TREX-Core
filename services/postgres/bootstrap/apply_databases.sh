#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

MODE="once"
case "${1:---once}" in
  --once)
    MODE="once"
    ;;
  --loop)
    MODE="loop"
    ;;
  *)
    echo "Usage: $0 [--once|--loop]" >&2
    exit 64
    ;;
esac

POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:?POSTGRES_USER is required}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
export PGPASSWORD="${PGPASSWORD:-$POSTGRES_PASSWORD}"
POSTGRES_BOOTSTRAP_CONTROL_DB="${POSTGRES_BOOTSTRAP_CONTROL_DB:-postgres}"
POSTGRES_BOOTSTRAP_SCHEMA="${POSTGRES_BOOTSTRAP_SCHEMA:-trex_bootstrap}"
POSTGRES_BOOTSTRAP_INTERVAL_SECONDS="${POSTGRES_BOOTSTRAP_INTERVAL_SECONDS:-30}"
POSTGRES_DATABASES_DIR="${POSTGRES_DATABASES_DIR:-/databases}"

if [[ ! "$POSTGRES_BOOTSTRAP_SCHEMA" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "POSTGRES_BOOTSTRAP_SCHEMA must be a simple SQL identifier; got '$POSTGRES_BOOTSTRAP_SCHEMA'" >&2
  exit 64
fi

log() {
  printf '[postgres-bootstrap] %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

psql_admin() {
  psql \
    --no-password \
    --no-psqlrc \
    -v ON_ERROR_STOP=1 \
    -X \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_BOOTSTRAP_CONTROL_DB" \
    "$@"
}

psql_target_db() {
  local db_name="$1"
  shift
  psql \
    --no-password \
    --no-psqlrc \
    -v ON_ERROR_STOP=1 \
    -X \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -d "$db_name" \
    "$@"
}

wait_for_postgres() {
  until pg_isready \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_BOOTSTRAP_CONTROL_DB" >/dev/null 2>&1; do
    log "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_BOOTSTRAP_CONTROL_DB} ..."
    sleep 2
  done
}

ensure_control_table() {
  psql_admin <<SQL
CREATE SCHEMA IF NOT EXISTS ${POSTGRES_BOOTSTRAP_SCHEMA};
CREATE TABLE IF NOT EXISTS ${POSTGRES_BOOTSTRAP_SCHEMA}.imported_databases (
  database_name text PRIMARY KEY,
  source_folder text NOT NULL,
  source_signature text NOT NULL,
  loaded_files text NOT NULL,
  status text NOT NULL,
  imported_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  notes text
);
SQL
}

database_exists() {
  local db_name="$1"
  local result
  result="$(
    psql_admin --set=db_name="$db_name" -Atq <<'SQL' || true
SELECT 1
FROM pg_database
WHERE datname = :'db_name';
SQL
  )"
  [[ "$result" == "1" ]]
}

recorded_status() {
  local db_name="$1"
  psql_admin --set=db_name="$db_name" -Atq <<SQL || true
SELECT status
FROM ${POSTGRES_BOOTSTRAP_SCHEMA}.imported_databases
WHERE database_name = :'db_name'
LIMIT 1;
SQL
}

has_success_record() {
  local db_name="$1"
  [[ "$(recorded_status "$db_name")" == "success" ]]
}

record_status() {
  local db_name="$1"
  local source_folder="$2"
  local source_signature="$3"
  local loaded_files="$4"
  local status="$5"
  local notes="${6:-}"

  psql_admin \
    --set=db_name="$db_name" \
    --set=source_folder="$source_folder" \
    --set=source_signature="$source_signature" \
    --set=loaded_files="$loaded_files" \
    --set=status="$status" \
    --set=notes="$notes" <<SQL
INSERT INTO ${POSTGRES_BOOTSTRAP_SCHEMA}.imported_databases (
  database_name,
  source_folder,
  source_signature,
  loaded_files,
  status,
  imported_at,
  updated_at,
  notes
)
VALUES (
  :'db_name',
  :'source_folder',
  :'source_signature',
  :'loaded_files',
  :'status',
  now(),
  now(),
  NULLIF(:'notes', '')
)
ON CONFLICT (database_name)
DO UPDATE
SET source_folder = EXCLUDED.source_folder,
    source_signature = EXCLUDED.source_signature,
    loaded_files = EXCLUDED.loaded_files,
    status = EXCLUDED.status,
    updated_at = now(),
    notes = EXCLUDED.notes;
SQL
}

create_database() {
  local db_name="$1"
  psql_admin --set=db_name="$db_name" <<'SQL'
SELECT format('CREATE DATABASE %I', :'db_name')\gexec
SQL
}

collect_loadable_files() {
  local db_dir="$1"
  find "$db_dir" \
    -mindepth 1 \
    -maxdepth 1 \
    -type f \
    \( -name '*.sql' -o -name '*.sql.gz' -o -name '*.dump' -o -name '*.tar' \) \
    -printf '%f\n' | LC_ALL=C sort
}

compute_signature() {
  local db_dir="$1"
  local items
  items="$(
    find "$db_dir" \
      -mindepth 1 \
      -maxdepth 1 \
      -type f \
      \( -name '*.sql' -o -name '*.sql.gz' -o -name '*.dump' -o -name '*.tar' \) \
      -printf '%f|%s|%TY-%Tm-%TdT%TH:%TM:%TS\n' | LC_ALL=C sort
  )"
  if [[ -z "$items" ]]; then
    printf 'empty'
  else
    printf '%s' "$items" | sha256sum | awk '{print $1}'
  fi
}

sql_file_contains_create_database() {
  local file_path="$1"
  case "$file_path" in
    *.sql)
      grep -Eq '^[[:space:]]*CREATE[[:space:]]+DATABASE[[:space:]]' "$file_path"
      ;;
    *.sql.gz)
      gzip -dc "$file_path" | grep -Eq '^[[:space:]]*CREATE[[:space:]]+DATABASE[[:space:]]'
      ;;
    *)
      return 1
      ;;
  esac
}

rewrite_sql_dump_for_bootstrap() {
  sed -E "s/ OWNER TO postgres;/ OWNER TO ${POSTGRES_USER};/g"
}

apply_full_sql_dump() {
  local file_path="$1"
  case "$file_path" in
    *.sql)
      rewrite_sql_dump_for_bootstrap < "$file_path" | psql_admin
      ;;
    *.sql.gz)
      gzip -dc "$file_path" | rewrite_sql_dump_for_bootstrap | psql_admin
      ;;
    *)
      log "Unsupported full SQL dump format ${file_path}"
      return 1
      ;;
  esac
}

apply_file() {
  local db_name="$1"
  local file_path="$2"
  case "$file_path" in
    *.sql)
      log "Applying SQL file $(basename "$file_path") into database ${db_name}"
      rewrite_sql_dump_for_bootstrap < "$file_path" | psql_target_db "$db_name"
      ;;
    *.sql.gz)
      log "Applying compressed SQL file $(basename "$file_path") into database ${db_name}"
      gzip -dc "$file_path" | rewrite_sql_dump_for_bootstrap | psql_target_db "$db_name"
      ;;
    *.dump|*.tar)
      log "Restoring archive $(basename "$file_path") into database ${db_name}"
      pg_restore \
        --no-password \
        --exit-on-error \
        --verbose \
        --no-owner \
        --no-privileges \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$db_name" \
        "$file_path"
      ;;
    *)
      log "Skipping unsupported file ${file_path}"
      ;;
  esac
}

process_database_folder() {
  local db_dir="$1"
  local db_name
  db_name="$(basename "$db_dir")"

  mapfile -t files < <(collect_loadable_files "$db_dir")
  if [[ ${#files[@]} -eq 0 ]]; then
    log "Skipping ${db_name}: no .sql, .sql.gz, .dump, or .tar files found"
    return 0
  fi

  local signature loaded_files
  signature="$(compute_signature "$db_dir")"
  loaded_files="$(printf '%s\n' "${files[@]}")"

  local full_dump_file=""
  if [[ ${#files[@]} -eq 1 ]] && sql_file_contains_create_database "$db_dir/${files[0]}"; then
    full_dump_file="${files[0]}"
  fi

  if database_exists "$db_name"; then
    local current_status
    current_status="$(recorded_status "$db_name")"

    if [[ "$current_status" == "success" ]]; then
      log "Skipping ${db_name}: database already exists and has a successful bootstrap record"
      return 0
    fi

    if [[ -n "$current_status" ]]; then
      log "Skipping ${db_name}: database already exists and bootstrap status is '${current_status}'"
      return 0
    fi

    log "Skipping ${db_name}: database already exists but was not created by this bootstrap controller"
    record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "external" \
      "Database exists without a bootstrap status record; left untouched."
    return 0
  fi

  if [[ -n "$full_dump_file" ]]; then
    log "Applying full SQL dump ${full_dump_file}; the dump will create database ${db_name}"
    if ! apply_full_sql_dump "$db_dir/$full_dump_file"; then
      record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "failed" \
        "Import failed while applying ${full_dump_file}"
      log "Import failed for ${db_name} while applying ${full_dump_file}"
      return 1
    fi

    record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "success" ""
    log "Database ${db_name} imported successfully"
    return 0
  fi

  log "Creating database ${db_name}"
  if ! create_database "$db_name"; then
    record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "failed" \
      "CREATE DATABASE failed"
    return 1
  fi

  local file_name
  for file_name in "${files[@]}"; do
    if ! apply_file "$db_name" "$db_dir/$file_name"; then
      record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "failed" \
        "Import failed while applying ${file_name}"
      log "Import failed for ${db_name} while applying ${file_name}"
      return 1
    fi
  done

  record_status "$db_name" "$db_dir" "$signature" "$loaded_files" "success" ""
  log "Database ${db_name} imported successfully"
}

run_once() {
  wait_for_postgres
  ensure_control_table

  if [[ ! -d "$POSTGRES_DATABASES_DIR" ]]; then
    log "Database source directory ${POSTGRES_DATABASES_DIR} does not exist; nothing to do"
    return 0
  fi

  local found_any=0
  while IFS= read -r -d '' db_dir; do
    found_any=1
    process_database_folder "$db_dir"
  done < <(find "$POSTGRES_DATABASES_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | LC_ALL=C sort -z)

  if [[ "$found_any" -eq 0 ]]; then
    log "No database folders found under ${POSTGRES_DATABASES_DIR}"
  fi
}

main() {
  if [[ "$MODE" == "once" ]]; then
    run_once
    return 0
  fi

  log "Starting in watch mode; rescanning ${POSTGRES_DATABASES_DIR} every ${POSTGRES_BOOTSTRAP_INTERVAL_SECONDS}s"
  while true; do
    if ! run_once; then
      log "A scan failed; the controller will retry on the next interval"
    fi
    sleep "$POSTGRES_BOOTSTRAP_INTERVAL_SECONDS"
  done
}

main "$@"
