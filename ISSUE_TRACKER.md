# Ruff Issue Tracker

Generated from `uv run ruff check . --statistics` on 2026-03-12.

Total issues: **419**
Total rule categories: **27**

## Todo

- [ ] `E501` `line-too-long` - 221 issues
- [ ] `ARG002` `unused-method-argument` - 48 issues
- [ ] `T201` `print` - 29 issues
- [ ] `N999` `invalid-module-name` - 18 issues
- [ ] `C408` `unnecessary-collection-call` - 15 issues
- [ ] `PLC0415` `import-outside-top-level` - 13 issues
- [ ] `F841` `unused-variable` - 10 issues
- [ ] `RET504` `unnecessary-assign` - 10 issues
- [ ] `SIM401` `if-else-block-instead-of-dict-get` - 9 issues
- [ ] `PLR0912` `too-many-branches` - 7 issues
- [ ] `RUF006` `asyncio-dangling-task` - 6 issues
- [ ] `DTZ005` `call-datetime-now-without-tzinfo` - 5 issues
- [ ] `PLR0915` `too-many-statements` - 4 issues
- [ ] `E711` `none-comparison` - 3 issues
- [ ] `RET503` `implicit-return` - 3 issues
- [ ] `SIM103` `needless-bool` - 3 issues
- [ ] `B007` `unused-loop-control-variable` - 2 issues
- [ ] `B024` `abstract-base-class-without-abstract-method` - 2 issues
- [ ] `RUF003` `ambiguous-unicode-character-comment` - 2 issues
- [ ] `RUF012` `mutable-class-default` - 2 issues
- [ ] `ARG001` `unused-function-argument` - 1 issue
- [ ] `F403` `undefined-local-with-import-star` - 1 issue
- [ ] `PLW0127` `self-assigning-variable` - 1 issue
- [ ] `PLW1510` `subprocess-run-without-check` - 1 issue
- [ ] `S603` `subprocess-without-shell-equals-true` - 1 issue
- [ ] `SIM118` `in-dict-keys` - 1 issue
- [ ] `T203` `p-print` - 1 issue

## Done

- [x] Total Ruff findings reduced from 475 to 419; active rule categories reduced from 37 to 27
- [x] Cleared rule categories: `A001`, `A002`, `B006`, `B008`, `N801`, `PLR1714`, `RUF002`, `SIM108`, `SIM113`, `SIM210`
- [x] `F401` `unused-import` - 2 issues fixed
- [x] `TREX_Core/devices/bess.py` cleanup reduced `T201`, `C408`, `SIM103`, and `SIM401`; `N999` moved to targeted Ruff ignores for legacy CamelCase modules
