# MQTT in TREX-Core

This document describes how MQTT is used in this repository today, based on the code under `TREX_Core/`.

The goal is to define the real broker contract well enough to run the broker as a separate Docker container and let TREX processes interoperate with it reliably.

## 1. High-level architecture

TREX uses MQTT as the message bus between these runtime roles:

- `sim_controller`: starts episodes, advances simulation time, and waits for all parties to finish each turn.
- `market`: receives bids/asks/meter data, performs matching and settlement, and publishes market-side events.
- `participant`: one process per participant; joins the market, reacts to round starts, submits bids/asks, meters energy, and acknowledges settlements.
- optional `policy_server`: only partially visible in this repo; the topic family exists, but the outbound request side is not implemented here.

All of these processes connect to the same broker endpoint from config:

- `config["server"]["host"]`
- `config["server"]["port"]`

The topic namespace is rooted at `market_id`.

Examples:

- `baseline/simulation/start_round`
- `baseline/start_round`
- `baseline/join_market/Building_1`
- `baseline/Building_1/market_info`

## 2. MQTT library and protocol assumptions

### Primary client stack

The active runtime clients all inherit from `TREX_Core/mqtt/base_gmqtt.py`, which uses `gmqtt`.

The current entrypoints are:

- `TREX_Core/participants/client.py`
- `TREX_Core/markets/client.py`
- `TREX_Core/sim_controller/client.py`

There is also a `TREX_Core/mqtt/base_paho.py`, but it is not wired into the current participant, market, or sim-controller entrypoints.

### Protocol version

`gmqtt` connects with MQTT 5.0 by default. TREX also uses MQTT 5 features:

- `user_property`
- `will_delay_interval`

For that reason, the broker container should support MQTT 5.

### Payload encoding

TREX commonly publishes Python `dict`, `list`, and `tuple` objects directly through `gmqtt`.

`gmqtt.Message` serializes:

- `dict`, `list`, `tuple` -> JSON text
- `int`, `float` -> ASCII text
- `str` -> UTF-8 text
- `None` -> empty payload

This matters because receivers usually do one of two things:

- treat the payload as a plain string
- call `json.loads(message["payload"])`

Important consequence:

- any Python tuple on publish becomes a JSON array on the wire and must be reconstructed as a tuple if code expects one

### Client IDs and sessions

Each MQTT client instance uses a random `cuid2` value as `client_id`.

Implications:

- client IDs are not stable across restarts
- durable broker-side sessions are not part of the design
- clean-session reconnect behavior is effectively ephemeral

### Auth and TLS

The current TREX MQTT code does not configure:

- MQTT username/password
- TLS
- client certificates

A containerized broker should therefore either:

- expose a plain internal MQTT listener, or
- require code changes if authentication/TLS is mandatory

## 3. Runtime topology and startup flow

The runner launches separate processes for market, sim controller, and each participant.

Relevant files:

- `TREX_Core/runner/runner.py`
- `TREX_Core/runner/make/market.py`
- `TREX_Core/runner/make/sim_controller.py`
- `TREX_Core/runner/make/participant.py`

Startup flow:

1. Config is loaded from `TREX_Core/configs/<name>.json`.
2. Participant `_default` settings are merged into each participant.
3. `market_id` is finalized.
4. The runner starts:
   - one market process
   - one sim-controller process
   - one participant process per participant
   - optional policy-server process(es)
5. Each process connects to the same broker.
6. The sim controller polls for market/participant presence over MQTT until the system is ready.
7. The sim controller starts the episode and time-step loop.

## 4. Config fields that affect MQTT

### Broker connection

In config files such as `TREX_Core/configs/_template.json`, `mqtt_test.json`, and `citylearn_test.json`, the MQTT broker is defined under:

```json
"server": {
  "host": "localhost",
  "port": 1883
}
```

These values are passed to:

- market client
- sim-controller client
- every participant client
- optional policy server processes

### `market.id` becomes the topic root

`config["market"]["id"]` is the topic prefix used by every MQTT topic.

Examples:

- `training`
- `baseline`
- `MicroTE3B`
- `training/0`

Important runtime behavior from `Runner.modify_config()` and `Runner.make_launch_list()`:

- if the simulation type is `baseline`, `training`, or `replay`, `market.id` is rewritten to that simulation type
- if parallel runs are enabled, the runner appends `/<idx>` to `market.id`

That means `market_id` can contain slashes and therefore create multi-level topic roots.

Example:

- `market_id = "training/0"` produces topics like `training/0/simulation/start_round`

This works because dispatching is based on the final topic segment, not a fixed topic depth.

### `market.close_steps`

`close_steps` affects the timing payloads exchanged over MQTT and the meaning of `last_settle` / `next_settle`, but not the broker itself.

### Market grid config

`market.grid` affects market-info payloads and grid settlement pricing:

- `price`
- `fee_ratio`
- optional `tou`

### Participant config

Participant config does not change the broker connection, but it changes:

- which participant IDs exist
- whether bids/asks are produced
- whether storage-related behavior appears in meter data
- whether algorithm/policy-server topics are relevant

### Database config

`config["database"]` is not part of the MQTT transport, but it is required for process startup. A broker-only container does not need database access, but the TREX app containers do.

## 5. Config and credential discovery

Config lookup is handled by:

- `TREX_Core/runner/runner.py`
- `TREX_Core/utils/db_utils.py`

Search roots include:

- `TREX_CORE_ROOT`
- current working directory
- current working directory plus `/TREX_Core`
- the package root

Database credentials are read from:

- `TREX_Core/configs/_credentials.json`

An example file exists at:

- `TREX_Core/configs/_credentials.json.example`

This matters for containerization because the TREX app containers need the config directory mounted or baked in. The MQTT container itself does not.

## 6. Topic dispatch model

The internal dispatch logic does not match full topic strings. Instead it:

1. splits the topic by `/`
2. scans segments from right to left
3. calls the first handler whose name matches a segment

Practical implications:

- the final topic segment is the most important routing token
- `market_id` may safely contain `/`
- topic suffix collisions are risky
- external integrations should preserve the existing suffix names exactly

Examples:

- topic `baseline/simulation/start_round` dispatches on `start_round`
- topic `baseline/join_market/Building_1` dispatches on `join_market`
- topic `training/0/Building_7/settled` dispatches on `settled`

## 7. Topic catalog

The tables below describe the MQTT contract that can be confirmed from this repo.

### 7.1 Presence, discovery, and registration

| Topic | Publisher | Subscriber | Payload on wire | Notes |
| --- | --- | --- | --- | --- |
| `{market_id}/join_market/{participant_id}` | participant | market | JSON object | Participant announces itself. Published with `retain=True`, `qos=1`, `user_property=[("to","^all")]`. |
| `{market_id}/{participant_id}/market_info` | market | participant | JSON object | Contains market id, market sid, and timezone. |
| `{market_id}/simulation/is_market_online` | sim_controller | market | empty string | Probe message. |
| `{market_id}/simulation/market_online` | market | sim_controller | market id string | Market presence response. |
| `{market_id}/simulation/is_participant_joined` | sim_controller | participants | empty string | Broadcast probe for participant presence. |
| `{market_id}/simulation/participant_joined` | participant | sim_controller | participant id string | Published only if participant believes it is already connected to the market. |

#### `join_market` payload

Published by `Participant.join_market()`:

```json
{
  "type": ["participant", "Residential"],
  "id": "Building_1",
  "sid": "baseline",
  "market_id": "baseline"
}
```

Notes:

- `type` starts as a Python tuple, but becomes a JSON array on the wire
- the market ultimately trusts the topic suffix for participant ID and overwrites `client_data["id"]` from `topic.split("/")[-1]`
- `sid` is an application routing label, not the MQTT client ID

#### `market_info` payload

Published by `markets/client.py`:

```json
{
  "id": "baseline",
  "sid": "baseline",
  "timezone": "America/Vancouver"
}
```

### 7.2 Simulation lifecycle topics

| Topic | Publisher | Subscriber | Payload on wire | Notes |
| --- | --- | --- | --- | --- |
| `{market_id}/simulation/start_episode` | sim_controller | market, participants | episode number | Broadcast at the start of each episode. |
| `{market_id}/simulation/start_round` | sim_controller | market | JSON object | Drives the market time step. |
| `{market_id}/start_round` | market | participants | JSON array | Participant-facing round-start payload after the market adds timing and market info. |
| `{market_id}/simulation/end_turn` | participant | sim_controller | participant id string | Signals that the participant finished its round. |
| `{market_id}/simulation/end_round` | market | sim_controller | market id string | Signals that the market finished its round. |
| `{market_id}/simulation/end_episode` | sim_controller | market, participants | JSON object | Broadcast between episodes. |
| `{market_id}/simulation/participant_ready` | participant | sim_controller | JSON object | Participant says it has finished end-of-episode work. |
| `{market_id}/simulation/market_ready` | market | sim_controller | market id string | Market says it has finished end-of-episode work. |
| `{market_id}/simulation/end_simulation` | sim_controller | market, participants | market id string | Broadcast when the whole simulation is complete. |

#### `simulation/start_round` payload from sim controller to market

```json
{
  "time": 1470024000,
  "duration": 3600,
  "update": true
}
```

#### `start_round` payload from market to participants

Published by `DoubleAuction.__start_round()`:

```json
[
  1470024000,
  3600,
  2,
  {
    "current_round": [0.1449, 0.069],
    "next_settle": [0.1449, 0.069]
  }
]
```

Meaning:

- `[0]`: round start timestamp
- `[1]`: round duration
- `[2]`: `close_steps`
- `[3]`: market info

#### `simulation/end_episode` payload

```json
{
  "episode": 1,
  "market_id": "baseline"
}
```

#### `simulation/participant_ready` payload

```json
{
  "Building_1": true
}
```

### 7.3 Market action topics

| Topic | Publisher | Subscriber | Payload on wire | Notes |
| --- | --- | --- | --- | --- |
| `{market_id}/bid` | participant | market | JSON array | Bid submission. |
| `{market_id}/ask` | participant | market | JSON array | Ask submission. |
| `{market_id}/{participant_id}/bid_ack` | market | participant | entry id string | Acknowledges accepted bid. |
| `{market_id}/{participant_id}/ask_ack` | market | participant | entry id string | Acknowledges accepted ask. |
| `{market_id}/{participant_id}/settled` | market | participant | JSON array | Settlement notification. |
| `{market_id}/settlement_delivered` | participant | market | JSON object | Participant acknowledges the settlement notification. |

#### Bid payload

Published by `Participant.bid()`:

```json
[
  "abc123",
  "Building_1",
  500,
  0.1449,
  [1470027600, 1470031200]
]
```

Meaning:

- `[0]`: entry id
- `[1]`: participant id
- `[2]`: quantity in Wh
- `[3]`: price in $/kWh
- `[4]`: delivery interval

#### Ask payload

Published by `Participant.ask()`:

```json
[
  "xyz789",
  "Building_2",
  400,
  0.069,
  [1470027600, 1470031200],
  "solar"
]
```

The extra final field is the energy source.

#### Settlement payload

Published by market settlement code:

```json
[
  "commit42",
  "abc123",
  "solar",
  250,
  [1470027600, 1470031200]
]
```

Meaning:

- `[0]`: commit id
- `[1]`: original bid or ask entry id
- `[2]`: source
- `[3]`: settled quantity
- `[4]`: delivery interval

#### Settlement-delivered payload

Published by the participant after local ledger processing:

```json
{
  "Building_1": "commit42"
}
```

The market waits until each settlement has been acknowledged twice:

- once by buyer
- once by seller

If those acknowledgements do not arrive, round completion can stall.

### 7.4 Metering and post-settlement accounting

| Topic | Publisher | Subscriber | Payload on wire | Notes |
| --- | --- | --- | --- | --- |
| `{market_id}/meter` | participant | market | JSON array | Participant meter/submeter payload for the current round. |
| `{market_id}/{participant_id}/extra_transaction` | market | participant | JSON object | Grid and financial residual transactions after settlement. |

#### Meter payload

Published by `Participant.__meter_energy()`:

```json
[
  "Building_1",
  [1470024000, 1470027600],
  {
    "generation": {
      "solar": 100,
      "bess": 0
    },
    "load": {
      "bess": {
        "solar": 0
      },
      "other": {
        "solar": 80,
        "bess": 0,
        "ext": 20
      }
    }
  }
]
```

#### Extra-transaction payload

Published by `DoubleAuction.__process_energy_exchange()`:

```json
{
  "time_delivery": [1470024000, 1470027600],
  "grid": {
    "buy": [
      [20, 0.1449]
    ],
    "sell": [
      [10, 0.069, "solar"]
    ]
  },
  "financial": {
    "buy": [
      {
        "quantity": 30,
        "energy_source": "solar",
        "settlement_price_sell": 0.069,
        "settlement_price_buy": 0.1449,
        "time_creation": 1470024000,
        "time_purchase": 1470027600
      }
    ],
    "sell": []
  }
}
```

The participant converts `time_delivery` back into a tuple and expands the simplified `grid.buy` / `grid.sell` lists into richer transaction records.

### 7.5 Optional algorithm / policy-server topics

These topics are visible in the repo, but the full request/response implementation is not present here.

Confirmed topics:

| Topic | Publisher | Subscriber | Payload on wire | Notes |
| --- | --- | --- | --- | --- |
| `{market_id}/algorithm/policy_server_ready` | external policy server | sim_controller | implementation-specific | Used by sim controller to know that policy service is ready. |
| `{market_id}/algorithm/{participant_id}/get_actions_return` | external policy server | participant | JSON object | Participant subscribes and forwards payload to `trader.get_actions_return()`. |
| `{market_id}/algorithm/{participant_id}/get_metadata_return` | external policy server | participant | JSON object | Participant subscribes and forwards payload to `trader.get_metadata_return()`. |
| `{market_id}/simulation/is_policy_server_online` | sim_controller | external policy server | empty string | Probe for policy server presence. |

What cannot be confirmed from this repo:

- the request topics used by a `policy_client` trader
- the exact payload schemas expected by the policy server

So if the policy server is containerized separately, its request-side MQTT contract must be taken from the missing policy-client implementation, not from this repo alone.

## 8. QoS, retain, and broker features actually used

### QoS

Subscriptions are usually created at QoS 2.

Publishes are mostly QoS 1, with a few QoS 2 messages:

- `market_info`: QoS 2
- `settled` in `MicroTE4`: QoS 2
- most other application messages: QoS 1

### Retained messages

`join_market` is published as a retained message.

Participants also configure a last will on the same topic:

- topic: `{market_id}/join_market/{participant_id}`
- payload: empty string
- `retain=True`
- `will_delay_interval=1`

This is effectively a retained tombstone for participant presence.

At normal simulation shutdown, participants also publish the same empty retained payload explicitly.

Broker implications:

- retained messages must be supported
- retained join-state cleanup should work correctly

### User properties

TREX publishes `user_property=[("to", value)]` on many messages, using values such as:

- `^all`
- `market_sid`
- `participant_sid`

Important detail:

- the TREX application code in this repo does not inspect the `to` property when consuming messages
- the property is therefore either informational, future-facing, or intended for an external routing layer

A broker container should preserve user properties, but the current code does not depend on broker-side filtering by `to`.

## 9. Round-completion dependencies

This is important for broker reliability and debugging.

The market considers a round complete only after all of the following are true:

1. all active participants submitted meter data
2. market-side matching finished
3. every settlement has been acknowledged by both counterparties

The sim controller advances the global time step only after:

1. every participant published `simulation/end_turn`
2. the market published `simulation/end_round`
3. if policy clients exist, the policy server published readiness

So lost messages or broken subscriptions can stall the simulation even if processes stay alive.

## 10. Known implementation caveats relevant to containerization

### A. MQTT 5 is the safe assumption

Because of `user_property` and `will_delay_interval`, use an MQTT 5 capable broker.

### B. Some subscribed topics appear unused

The following subscriptions exist but do not have a meaningful handler path in the current code:

- participant subscribes to `{market_id}`
- participant subscribes to `{market_id}/{participant_id}`
- market subscribes to `{market_id}`
- market subscribes to `{market_id}/{market_id}`

Treat them as legacy or placeholder topics, not required integration points.

### C. `participant_disconnected` is incomplete

`sim_controller/client.py` contains a handler for `participant_disconnected`, but:

- there is no matching subscription in `SUBS`
- no publisher for that topic was found in this repo

### D. `extra_transaction` de-dup code looks incomplete

`participants/client.py` contains de-dup logic based on:

- `self._seen_extra_pids`
- `self._pid_cache_seconds`

Those attributes are not initialized in that file. If that path executes with a `packet_id`, it may fail unless initialized elsewhere.

### E. The Paho backbone is not the active path

`TREX_Core/mqtt/base_paho.py` exists, but current runtime code imports `base_gmqtt.py`.

If the broker/container design is based on current code, optimize for `gmqtt`, not the Paho adapter.

## 11. What the broker container must provide

At minimum:

- one reachable MQTT endpoint on the configured host/port
- MQTT 5 support
- retained message support
- QoS 1 and QoS 2 support
- stable connectivity between all TREX app containers and the broker

Recommended operational properties:

- topic-level isolation by `market_id`
- logs for connect/disconnect/publish/subscribe during bring-up
- persistence for retained messages if the broker may restart during a run

Not required by the current code:

- broker authentication
- TLS
- durable client sessions keyed by stable client IDs
- broker-side interpretation of the `to` user property

## 12. Practical Docker notes

For a Dockerized deployment, the clean separation is:

- one broker container
- one or more TREX app containers
- optionally separate policy-server containers
- separate database container(s) if needed

Set `config["server"]["host"]` to the broker container DNS name on the Docker network.

Example conceptually:

- broker service name: `mqtt`
- TREX config: `"server": { "host": "mqtt", "port": 1883 }`

If you run multiple simulations in parallel, remember that:

- they may share the same broker
- they are separated by `market_id`
- the runner may append `/<idx>` to `market_id`, which becomes part of the topic hierarchy

## 13. Files to read when modifying the MQTT contract

Primary files:

- `TREX_Core/mqtt/base_gmqtt.py`
- `TREX_Core/participants/client.py`
- `TREX_Core/participants/base.py`
- `TREX_Core/participants/ledger.py`
- `TREX_Core/markets/client.py`
- `TREX_Core/markets/base/DoubleAuction.py`
- `TREX_Core/markets/MicroTE4.py`
- `TREX_Core/sim_controller/client.py`
- `TREX_Core/sim_controller/sim_controller.py`
- `TREX_Core/runner/runner.py`
- `TREX_Core/runner/make/*.py`

Config examples:

- `TREX_Core/configs/_template.json`
- `TREX_Core/configs/mqtt_test.json`
- `TREX_Core/configs/citylearn_test.json`
- `TREX_Core/configs/citylearn_test3.json`
