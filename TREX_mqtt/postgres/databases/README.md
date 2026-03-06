# Folder-driven database imports

Each direct child directory under `TREX_mqtt/postgres/databases/` declares one PostgreSQL database.

Examples:

```text
TREX_mqtt/postgres/databases/
  citylearn_2022/
    001-schema.sql
    010-data.sql
  citylearn_test3/
    dump.sql
  some_other_db/
    restore.dump
```

Rules:

- The folder name becomes the target database name.
- Supported file types are `.sql`, `.sql.gz`, `.dump`, and `.tar`.
- Files are applied in lexical filename order.
- `.sql` and `.sql.gz` files are executed with `psql` against the already-created database.
- `.dump` and `.tar` files are restored with `pg_restore` into the already-created database.
- The bootstrap controller only imports a folder when the target database does **not** already exist.
- Re-running the controller is safe; existing databases are left untouched.

Operational notes:

- To add a new database, create a new folder here, add the restore files, and wait for the bootstrap controller scan interval or run the manual bootstrap helper.
- If you intentionally want to re-import a database, drop that database first; the controller will recreate and reload it on the next scan.
- If an import failed partway through and the database now exists, the controller will skip it on later scans to avoid destructive replay. Drop the partially imported database and rerun the bootstrap if you want a clean retry.
- Plain `.sql` files should contain schema/data for the target database, not a conflicting `CREATE DATABASE`/`\connect` flow. For full-database restores, custom-format or tar-format dumps are preferred.
