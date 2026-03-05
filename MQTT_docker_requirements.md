# MQTT Docker Requirements for TREX-Core

This document defines the requirements for packaging the TREX-Core MQTT layer as Docker-based infrastructure.

It is intended to be consumed by an AI agent that will design and/or implement the containerized MQTT stack.

This is a requirements document, not an implementation guide. It defines what must be true for the resulting design to be acceptable.

## 1. Purpose

The Dockerized MQTT design must provide a reliable message bus for TREX-Core runtime processes:

- `market`
- `sim_controller`
- `participant` processes
- optional external policy-server processes

The design must preserve the actual application contract already implemented in this repository. The broker layer must not reinterpret or reshape application messages.

## 2. Source-of-truth assumptions from the codebase

The following assumptions are derived from the current code and are mandatory design inputs.

### 2.1 MQTT client behavior already in TREX-Core

- The active MQTT client stack uses `gmqtt`, not the inactive `paho` backbone.
- TREX currently relies on MQTT 5 features, specifically:
  - `user_property`
  - `will_delay_interval`
- TREX publishes Python `dict`, `list`, and `tuple` objects through `gmqtt`, which are serialized as JSON text on the wire.
- TREX also publishes plain strings, integers, floats, and empty payloads.

### 2.2 Topic model

- All topics are rooted at `market_id`.
- `market_id` may contain `/`.
- Topic dispatch inside TREX depends on the final topic segment, not full fixed-depth topic paths.
- The broker design must therefore permit arbitrary multi-level topic roots and must not assume a fixed namespace depth.

### 2.3 QoS and retained-message behavior

- TREX uses QoS 1 heavily.
- TREX uses QoS 2 in some subscriptions and selected publishes.
- TREX uses retained `join_market` messages.
- Participants use MQTT last will on the same retained `join_market` topic to publish an empty retained payload when a participant disappears.

### 2.4 Current security posture in code

- TREX MQTT clients do not currently configure:
  - username/password
  - TLS
  - client certificates
- Therefore the initial Docker design must work on a trusted internal Docker network without requiring application code changes.

### 2.5 Runtime sensitivity

- The simulation can stall if MQTT messages are dropped, delayed, or not delivered to the expected subscribers.
- Round completion depends on:
  - participant meter messages
  - participant `end_turn` messages
  - market `end_round` messages
  - settlement acknowledgements from both counterparties
- The MQTT design is therefore part of the simulation correctness boundary, not just infrastructure.

## 3. Scope

The design must cover:

- the MQTT broker container
- Docker networking for TREX processes and the broker
- persistent storage requirements for broker state
- broker configuration management
- health monitoring
- metrics collection with Prometheus
- log collection expectations
- operational restart behavior
- validation and acceptance criteria

The design does not need to redesign TREX application code unless explicitly required by a future task.

## 4. Primary design goals

The containerized MQTT solution must satisfy all of the following goals:

- preserve TREX MQTT behavior without changing the application-level message contract
- keep the broker reachable by all TREX runtime containers through stable Docker DNS
- support MQTT 5 correctly
- support retained messages and QoS 1/QoS 2 correctly
- provide operational visibility through Prometheus-compatible metrics
- be simple enough to run locally in Docker Compose and later move to a more managed deployment
- avoid introducing security requirements that the current code cannot satisfy

## 5. Mandatory functional requirements

### 5.1 Broker protocol support

The broker implementation must:

- support MQTT 5.0
- support TCP MQTT on the configured port
- support wildcard subscriptions used by TREX
- support retained messages
- support QoS 0, QoS 1, and QoS 2
- support MQTT will messages with MQTT 5 properties
- preserve MQTT user properties end to end

The broker implementation must not:

- strip MQTT 5 user properties
- disable retained messages
- rewrite topic names
- transform payload bodies

### 5.2 Connection contract

The design must allow TREX containers to connect using:

- `server.host`
- `server.port`

The reference service name inside Docker should be:

- `mqtt`

This means the default container-network contract should permit TREX configs such as:

```json
"server": {
  "host": "mqtt",
  "port": 1883
}
```

### 5.3 Topic and payload neutrality

The broker layer must treat TREX topics and payloads as opaque application data.

It must not:

- enforce special meaning on `user_property=("to", ...)`
- require fixed topic depth
- assume `market_id` is a single topic segment
- require schema validation inside the broker

### 5.4 Retained-state handling

The design must correctly support retained presence behavior for:

- `{market_id}/join_market/{participant_id}` retained join payloads
- retained empty tombstones written through last will or explicit disconnect logic

If broker persistence is enabled, retained state must survive broker restart.

If broker persistence is disabled, the design must explicitly state that retained join-state is lost on restart and that TREX processes must reconnect and republish.

### 5.5 Restart and failure behavior

The broker deployment must define behavior for:

- broker restart
- app-container restart
- transient disconnects
- Docker daemon restart

At minimum:

- the broker container must restart automatically on failure
- TREX app containers must be able to reconnect to the broker after broker restart
- the design must not require manual broker reconfiguration after restart

## 6. Non-functional requirements

### 6.1 Reliability

The design must prioritize message delivery reliability over maximum throughput.

The design should:

- prefer predictable broker behavior over advanced clustering features
- avoid plugins or routing layers that can silently alter MQTT semantics
- keep the first implementation single-broker unless clustering is explicitly required

### 6.2 Performance

The design must be adequate for:

- one market process
- one sim-controller process
- many participant processes
- optional policy-server processes
- potentially multiple parallel simulations separated by `market_id`

The design must expose configurable limits for:

- maximum connections
- queued/inflight messages
- retained-message storage
- persistence storage location
- file-descriptor limits if relevant

### 6.3 Operability

The broker container must:

- run without interactive setup
- be configurable through mounted config files and/or environment variables
- write logs to stdout/stderr for Docker-native collection
- support health checking

### 6.4 Simplicity

The first acceptable design should prefer:

- a single broker container
- a single persistent volume
- one Prometheus scrape target for broker metrics
- minimal extra moving parts

If an exporter sidecar is required, it is acceptable, but only if the broker cannot expose Prometheus metrics natively.

## 7. Broker selection requirements

The chosen broker must satisfy all mandatory functional requirements.

The preferred broker choice should satisfy Prometheus monitoring in one of these two ways:

1. Native Prometheus metrics endpoint.
2. A well-supported Prometheus exporter or sidecar.

Broker-selection decision criteria must include:

- correct MQTT 5 support
- retained-message support
- QoS 1/QoS 2 correctness
- operational maturity
- Docker suitability
- Prometheus integration quality
- configuration simplicity

The design may recommend a specific broker, but it must justify the choice against the above criteria.

## 8. Docker deployment requirements

### 8.1 Network topology

The design must place the following on a shared Docker network:

- broker container
- TREX runtime containers
- Prometheus container

The broker listener should be:

- reachable by service DNS name from the TREX containers
- optionally exposed to the host for local development
- not exposed publicly by default

### 8.2 Persistent volumes

The design must define persistent storage for broker state when persistence is enabled.

That storage must cover, as applicable:

- retained messages
- durable broker metadata
- local broker persistence files

The design should define distinct logical locations for:

- config
- data
- logs, if the broker writes file logs

### 8.3 Health checks

The design must include a broker health check.

Acceptable health-check strategies include:

- TCP connect check on the MQTT listener
- broker-native status command
- HTTP management/metrics endpoint check, if available

The health check must be suitable for:

- Docker Compose
- automated restart policy handling
- Prometheus alert correlation

### 8.4 Resource policy

The design must specify configurable resource settings for:

- CPU
- memory
- disk usage

The design should recommend conservative defaults and allow overrides.

## 9. Security requirements

### 9.1 Baseline deployment security

The initial deployment may assume a trusted internal Docker network because current TREX code does not provide MQTT auth or TLS configuration.

However, the design must still:

- avoid exposing the broker publicly by default
- avoid default credentials committed in source control
- keep future auth/TLS enablement possible without redesigning the whole stack

### 9.2 Future-ready security hooks

The design should provide a path for later enabling:

- username/password authentication
- TLS encryption
- secret injection through Docker secrets or environment variables
- topic ACLs, if TREX is later updated to support them cleanly

### 9.3 Container hardening

The broker container should:

- run as non-root where supported
- use read-only filesystem where practical
- mount only the directories it needs
- avoid unnecessary Linux capabilities

## 10. Observability requirements

The MQTT Docker design must include observability for:

- broker liveness
- broker load
- connection behavior
- message throughput
- queue/backlog pressure
- persistence health
- container resource usage

Observability must include:

- Prometheus metrics
- broker/application logs
- health status

## 11. Prometheus monitoring requirements

### 11.1 Metrics exposure

The design must expose broker metrics in a Prometheus-compatible way.

This can be done through:

- a native `/metrics` endpoint on the broker, or
- a dedicated exporter container/sidecar

The Prometheus scrape path, scrape port, and scrape interval must be explicitly defined in the design.

### 11.2 Required metric categories

The monitoring design must include metrics for the following categories.

#### Broker availability

- broker up/down status
- broker uptime
- restart count or equivalent reset detection
- listener availability

#### Connections and sessions

- current client connections
- connection rate
- disconnect rate
- rejected connections, if available
- session count, if the broker exposes it

#### Subscription and message flow

- publish rate
- receive rate
- subscribe rate, if available
- bytes in
- bytes out
- dropped message count, if available
- retained-message count, if available

#### QoS and delivery pressure

- inflight message count
- queued message count
- retry/redelivery count, if available
- unacknowledged QoS message count, if available

#### Persistence and storage

- broker persistence health
- persistence errors
- disk usage for broker data
- retained/persistence store size, if available

#### Container/process health

- CPU usage
- memory usage
- file descriptors, if available
- container restart count

### 11.3 Recommended labels

Metrics should include or be enrichable with labels that make simulations distinguishable operationally.

Recommended labels:

- `service`
- `instance`
- `environment`
- `broker_node`
- `listener`

If topic-level labels are available, they must be used carefully. High-cardinality labels must be avoided.

The design must not rely on per-topic metrics that explode cardinality when many `market_id` values or participant IDs exist.

### 11.4 Prometheus scrape requirements

The Prometheus configuration must:

- scrape the broker metrics target
- scrape the exporter if an exporter is used
- define sane scrape intervals for local development and simulation runs

Recommended default:

- scrape interval: `15s`
- evaluation interval: `15s`

The design may use different values if justified.

## 12. Alerting requirements

The monitoring design must define alert rules, even if Alertmanager is not implemented in the first task.

At minimum, the design must specify alerts for:

- broker unavailable
- metrics endpoint unavailable
- sudden connection loss or connection count collapse
- high reconnect churn
- persistent message drops or delivery failures
- high inflight or queued message counts
- persistence/disk problems
- abnormal broker restart frequency

Recommended initial alert set:

- `MQTTBrokerDown`
- `MQTTMetricsEndpointDown`
- `MQTTConnectionCountTooLow`
- `MQTTReconnectStorm`
- `MQTTDroppedMessagesDetected`
- `MQTTQueueBacklogHigh`
- `MQTTPersistenceError`
- `MQTTBrokerDiskUsageHigh`
- `MQTTBrokerRestartingFrequently`

Each alert definition in the final implementation should specify:

- condition
- threshold
- duration
- severity
- remediation hint

## 13. Logging requirements

The design must define a logging approach for the broker.

Minimum requirements:

- broker logs must be emitted to stdout/stderr or otherwise collected by Docker-native tooling
- logs must include connect/disconnect and error events
- logs must be timestamped

The design should also cover:

- optional debug logging for publish/subscribe troubleshooting
- how to enable verbose logging temporarily
- how to avoid logging full payloads by default if that creates noise or data leakage

## 14. Configuration management requirements

The design must define where broker configuration lives and how it is injected.

It must cover:

- broker config file path
- persistence path
- metrics configuration path, if separate
- Docker Compose service configuration
- Prometheus scrape configuration

The design should support:

- local development overrides
- environment-specific overrides
- explicit ports and volumes

## 15. Acceptance criteria

The Dockerized MQTT design is acceptable only if all of the following are true.

### 15.1 Connectivity validation

- TREX containers can resolve the broker by Docker service name.
- TREX market, sim-controller, and participant processes can all connect without code changes other than config values.

### 15.2 Protocol validation

- MQTT 5 user properties are accepted and preserved.
- QoS 1 and QoS 2 flows complete successfully.
- retained `join_market` messages behave correctly.
- last-will tombstone behavior works correctly.

### 15.3 Simulation validation

- a TREX simulation can start and complete using the Dockerized broker.
- participants can join the market and receive `market_info`.
- round progression works end to end.
- settlement acknowledgements and meter messages are delivered reliably enough for the simulation not to stall.

### 15.4 Monitoring validation

- Prometheus can scrape the broker metrics target.
- broker health is visible in Prometheus.
- connection and message-flow metrics are visible.
- at least one test alert can be evaluated successfully.

### 15.5 Restart validation

- restarting the broker container results in predictable reconnection behavior.
- the behavior of retained data after restart is documented and matches configuration.

## 16. Required deliverables for the implementing AI agent

The implementing AI agent should produce, at minimum:

- Docker Compose or equivalent container orchestration definition
- broker configuration file
- persistent-volume definition
- Prometheus configuration for scraping broker metrics
- alert rule file
- short operator-facing runbook or README

If the chosen broker requires an exporter, the agent should also produce:

- exporter container definition
- exporter configuration

## 17. Decision points the implementation must document

The final implementation must explicitly document the decisions made for:

- chosen MQTT broker and why
- whether persistence is enabled
- whether a metrics exporter is required
- how health checks are performed
- which ports are exposed internally and externally
- which volumes are required
- how future auth/TLS would be added

## 18. Recommended implementation posture

The first implementation should optimize for correctness and observability, not clustering or maximum scale.

Preferred characteristics for the first implementation:

- single broker
- single Docker network
- persistent retained-state support
- Prometheus integration from day one
- minimal operational complexity

## 19. Out of scope for the first implementation

Unless explicitly requested, the first implementation does not need to include:

- multi-node broker clustering
- Internet-facing broker exposure
- production-grade certificate automation
- broker-side business-logic plugins
- broker-side topic rewriting
- redesign of TREX application topics

## 20. Relationship to the existing MQTT contract

This requirements document must be interpreted together with:

- `MQTT.md`

`MQTT.md` describes the current TREX MQTT behavior.

This file defines the infrastructure and monitoring requirements for running that behavior inside Docker.
