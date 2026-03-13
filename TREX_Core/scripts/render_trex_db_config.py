#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render TREX database config JSON files and _credentials.json"
            " from environment variables."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a TREX config JSON file to update in place."
        " May be provided multiple times.",
    )
    parser.add_argument(
        "--credentials",
        help="Path to TREX_Core/configs/_credentials.json."
        " Defaults to a sibling of the first config file.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def resolve_credentials_path(first_config: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return first_config.parent / "_credentials.json"


def resolve_db_settings(existing_db: dict[str, Any]) -> dict[str, Any]:
    host = first_env("TREX_DB_HOST", "POSTGRES_HOST") or existing_db.get("host")
    port_raw = first_env("TREX_DB_PORT") or str(existing_db.get("port", 5432))
    connector = first_env("TREX_DB_CONNECTOR") or existing_db.get(
        "connector", "postgresql+psycopg"
    )
    profiles_db = first_env("TREX_PROFILES_DB") or existing_db.get("profiles_db")

    if host in (None, ""):
        raise SystemExit(
            "Missing TREX database host. Set TREX_DB_HOST in the environment"
            " or keep a host value in the JSON file."
        )

    try:
        port = int(port_raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid TREX_DB_PORT value: {port_raw!r}") from exc

    if profiles_db in (None, ""):
        raise SystemExit(
            "Missing TREX profiles database. Set TREX_PROFILES_DB in the environment"
            " or keep database.profiles_db in the JSON file."
        )

    return {
        "host": host,
        "port": port,
        "connector": connector,
        "profiles_db": profiles_db,
    }


def resolve_credentials() -> dict[str, str]:
    username = first_env("TREX_DB_USERNAME", "POSTGRES_USER")
    password = first_env("TREX_DB_PASSWORD", "POSTGRES_PASSWORD")

    if username is None or username == "":
        raise SystemExit(
            "Missing TREX database username."
            " Set TREX_DB_USERNAME or POSTGRES_USER in the environment."
        )
    if password is None or password == "":
        raise SystemExit(
            "Missing TREX database password."
            " Set TREX_DB_PASSWORD or POSTGRES_PASSWORD in the environment."
        )

    return {"username": username, "password": password}


def update_config(config_path: Path, db_settings: dict[str, Any]) -> None:
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Expected top-level object in {config_path}, got {type(payload).__name__}"
        )

    database_block = payload.setdefault("database", {})
    if not isinstance(database_block, dict):
        raise SystemExit(f"Expected 'database' to be an object in {config_path}")

    database_block.update(db_settings)
    atomic_write_json(config_path, payload)


def main() -> int:
    args = parse_args()
    config_paths = [Path(path) for path in args.config]
    credentials_path = resolve_credentials_path(config_paths[0], args.credentials)

    first_payload = read_json(config_paths[0])
    existing_db = (
        first_payload.get("database", {}) if isinstance(first_payload, dict) else {}
    )
    if existing_db is None:
        existing_db = {}
    if not isinstance(existing_db, dict):
        raise SystemExit(f"Expected 'database' to be an object in {config_paths[0]}")

    db_settings = resolve_db_settings(existing_db)
    credentials = resolve_credentials()

    for config_path in config_paths:
        update_config(config_path, db_settings)
        log.info(f"Updated database block in {config_path}")

    atomic_write_json(credentials_path, credentials)
    log.info(f"Wrote credentials to {credentials_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
