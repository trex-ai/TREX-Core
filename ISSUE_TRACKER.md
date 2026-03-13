# Ruff Issue Tracker

Generated from `uv run ruff check . --statistics` on 2026-03-13.

Total issues: **375**
Total rule categories: **23**

## Todo

- [ ] `E501` `line-too-long` - 206 issues
- [ ] `ARG002` `unused-method-argument` - 39 issues
- [ ] `T201` `print` - 25 issues
- [ ] `N999` `invalid-module-name` - 17 issues
- [ ] `C408` `unnecessary-collection-call` - 13 issues
- [ ] `PLC0415` `import-outside-top-level` - 12 issues
- [ ] `F841` `unused-variable` - 9 issues
- [ ] `SIM401` `if-else-block-instead-of-dict-get` - 8 issues
- [ ] `PLR0912` `too-many-branches` - 7 issues
- [ ] `RET504` `unnecessary-assign` - 7 issues
- [ ] `RUF006` `asyncio-dangling-task` - 6 issues
- [ ] `DTZ005` `call-datetime-now-without-tzinfo` - 4 issues
- [ ] `PLR0915` `too-many-statements` - 4 issues
- [ ] `E711` `none-comparison` - 3 issues
- [ ] `RET503` `implicit-return` - 3 issues
- [ ] `B007` `unused-loop-control-variable` - 2 issues
- [ ] `B024` `abstract-base-class-without-abstract-method` - 2 issues
- [ ] `RUF003` `ambiguous-unicode-character-comment` - 2 issues
- [ ] `SIM103` `needless-bool` - 2 issues
- [ ] `PLW0127` `self-assigning-variable` - 1 issue
- [ ] `PLW1510` `subprocess-run-without-check` - 1 issue
- [ ] `S603` `subprocess-without-shell-equals-true` - 1 issue
- [ ] `T203` `p-print` - 1 issue

## Done

- [x] Total Ruff findings reduced from 475 to 375; active rule categories reduced from 37 to 23
- [x] Cleared rule categories: `A001`, `A002`, `ARG001`, `B006`, `B008`, `F403`, `N801`, `PLR1714`, `RUF002`, `RUF012`, `SIM108`, `SIM113`, `SIM118`, `SIM210`
- [x] `F401` `unused-import` - 2 issues fixed
- [x] `TREX_Core/devices/bess.py` cleanup reduced `T201`, `C408`, `SIM103`, and `SIM401`; `N999` moved to targeted Ruff ignores for legacy CamelCase modules
- [x] `TREX_Core/utils/records.py` cleanup fixed the file-level Ruff failures and helped reduce `ARG002`, `C408`, `DTZ005`, `E501`, and `SIM401`
