# https://stackoverflow.com/questions/30778015/how-to-increase-the-max-connections-in-postgres

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import commentjson
import databases
import sqlalchemy
from sqlalchemy import Column, MetaData, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists


def _iter_config_roots(root_dir: str = "") -> Iterator[Path]:
    roots: list[Path] = []
    if root_dir:
        supplied_root = Path(root_dir).expanduser().resolve()
        roots.extend((supplied_root, supplied_root / "TREX_Core"))
    else:
        env_root = os.environ.get("TREX_CORE_ROOT", "").strip()
        if env_root:
            env_root_path = Path(env_root).expanduser().resolve()
            roots.extend((env_root_path, env_root_path / "TREX_Core"))

        cwd = Path.cwd().resolve()
        roots.extend((cwd, cwd / "TREX_Core"))
        roots.append(Path(__file__).resolve().parents[1])

    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        yield root


def get_credentials(root_dir: str = "") -> Any:
    def _load_json_file(file_path: Path) -> Any:
        with open(file_path) as f:
            return commentjson.load(f)

    for base_dir in _iter_config_roots(root_dir):
        credentials_file = base_dir / "configs" / "_credentials.json"
        if credentials_file.is_file():
            return _load_json_file(credentials_file)
    return None


def make_db_str(
    credentials: dict, db_config: dict, db_name: str = "", table_name: str = ""
) -> str:
    connector = db_config["connector"]
    host = db_config["host"]
    port = db_config["port"]
    db_str = f"{connector}://{credentials['username']}:{credentials['password']}@{host}:{port}"
    if db_name:
        db_str += f"/{db_name}"
    if table_name:
        db_str += f"/{table_name}"
    return db_str


def create_db(db_string, engine=None):
    if not engine:
        engine = create_engine(db_string)
    if not database_exists(engine.url):
        create_database(engine.url)
    return database_exists(engine.url)


async def dump_data(data, db_string, table, existing_connection=None):
    """Insert multiple records into a database table efficiently.

    Args:
        data: List of records to insert
        db_string: Database connection string
        table: SQLAlchemy Table object
        existing_connection: Optional existing database connection to reuse
    """
    if not data:
        return  # Short-circuit for empty data

    # Use existing connection if provided, otherwise create a new one
    if existing_connection:
        async with existing_connection.transaction():
            query = table.insert()
            await existing_connection.execute_many(query, data)
        return

    # Create a new connection
    async with databases.Database(db_string) as db:
        async with db.transaction():
            query = table.insert()
            await db.execute_many(query, data)
        return


def get_table(db_string, table_name, engine=None):
    if not engine:
        engine = create_engine(db_string)

    if not sqlalchemy.inspect(engine).has_table(table_name):
        return None

    metadata = MetaData()
    return sqlalchemy.Table(table_name, metadata, autoload_with=engine)


def get_table_len(db_string, table, engine=None):
    if not engine:
        engine = create_engine(db_string)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        return session.query(table).count()


def drop_table(db_string, table_name, engine=None):
    if not engine:
        engine = create_engine(db_string)
    table = get_table(db_string, table_name, engine)
    if table is not None:
        table.drop(engine)


async def create_market_table(db_string, table_name=None, engine=None):
    if not engine:
        engine = create_engine(db_string)
    if not database_exists(engine.url):
        create_db(db_string)

    if sqlalchemy.inspect(engine).has_table(table_name):
        return None

    meta = MetaData()
    table = sqlalchemy.Table(
        table_name,
        meta,
        Column("id", sqlalchemy.Integer, primary_key=True),
        Column("quantity", sqlalchemy.Integer),
        Column("seller_id", sqlalchemy.String),
        Column("buyer_id", sqlalchemy.String),
        Column("energy_source", sqlalchemy.String),
        Column("settlement_price_sell", sqlalchemy.Float),
        Column("settlement_price_buy", sqlalchemy.Float),
        Column("fee_ask", sqlalchemy.Float),
        Column("fee_bid", sqlalchemy.Float),
        Column("time_creation", sqlalchemy.Integer),
        Column("time_purchase", sqlalchemy.Integer),
        Column("time_consumption", sqlalchemy.Integer),
    )
    table.create(engine, checkfirst=True)
    return True


async def create_table(db_string, table, engine=None):
    if not engine:
        engine = create_engine(db_string)
    if not database_exists(engine.url):
        create_db(db_string)

    if sqlalchemy.inspect(engine).has_table(table.name):
        return None
    table.create(engine, checkfirst=True)
    return True


async def update_metadata(db_string, generation, update_dict):
    db = databases.Database(db_string)
    await db.connect()
    md_table = get_table(db_string, "metadata")
    async with db.transaction():
        md = await db.fetch_one(
            md_table.select(md_table.c.generation == generation, for_update=True)
        )
        if md is None:
            await db.disconnect()
            return False

        metadata = md["data"]

        # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def update(d: dict[str, Any], u: Mapping[str, Any]) -> dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        metadata = update(metadata, update_dict)

        await db.execute(
            md_table.update()
            .where(md_table.c.generation == generation)
            .values(data=metadata)
        )
    await db.disconnect()
    return True
