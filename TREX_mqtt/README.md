# TREX-Core MQTT Docker stack

This bundle implements a first-cut, correctness-first MQTT stack for TREX-Core with these decisions:

- **Broker**: single-node **EMQX 5.x**
- **Broker DNS name inside Docker**: `mqtt`
- **Persistence**: enabled by default through `./data/emqx`
- **Broker metrics**: native EMQX Prometheus pull endpoints
- **Extra observability**: cAdvisor for container CPU/memory/start-time and a tiny read-only directory-size exporter for EMQX data/log paths
- **Health check**: `emqx ctl status`
- **Host exposure**: loopback only by default (`127.0.0.1`), not public

## Why this shape

This stack is optimized for TREX correctness before scale:

1. It keeps MQTT semantics intact. The broker is used as a plain broker only: no topic rewrite, no payload transformation, no schema validation, and no broker-side business logic.
2. It keeps the first deployment single-node. That avoids clustering complexity and makes restart, persistence, and debugging behavior easier to reason about.
3. It enables persistence from day one so retained `join_market` state survives broker restarts as long as `./data/emqx` is preserved.
4. It exposes broker health and message-flow counters immediately through Prometheus.

## File layout

- `docker-compose.yml` — broker, Prometheus, cAdvisor, and the volume metrics helper
- `emqx/base.hocon` — broker baseline configuration
- `prometheus/prometheus.yml` — scrape targets and intervals
- `prometheus/alerts/mqtt-alerts.yml` — initial alert rules
- `ops/volume_metrics_exporter.py` — read-only helper that reports data/log directory size
- `.env.example` — overridable defaults and image tags

## Bring-up

```bash
cp .env.example .env
docker compose up -d
docker compose ps
docker compose logs -f mqtt
```

Prometheus will be available on:

```text
http://127.0.0.1:9090
```

EMQX Dashboard and API will be available on:

```text
http://127.0.0.1:18083
```

The MQTT listener will be available on:

```text
mqtt:1883
```

from other containers on the shared `trex` Docker network, and on:

```text
127.0.0.1:1883
```

from the host.

## TREX connection contract

The expected TREX-side broker config stays:

```json
{
  "server": {
    "host": "mqtt",
    "port": 1883
  }
}
```

No application-level topic, payload, or MQTT property rewrites are introduced.

## Persistence and restart behavior

### Broker restart

- Broker state lives under `./data/emqx`.
- Retained messages and local broker metadata survive a broker container restart as long as that directory is preserved.
- The broker comes back with the same Docker DNS name and the same node name (`emqx@mqtt`).

### App-container restart

- TREX containers reconnect to the same `mqtt:1883` endpoint.
- No manual broker reconfiguration is needed.

### Docker daemon restart

- All services use `restart: unless-stopped`.
- After the Docker engine returns, the stack comes back with the same bind mounts, service name, and broker identity.

### Persistence disabled

If you remove or replace `./data/emqx`, retained `join_market` state is lost. In that mode, TREX processes must reconnect and republish retained state.

## Health checks

The broker health check uses:

```bash
/opt/emqx/bin/emqx ctl status
```

This is suitable for Docker Compose restarts and easy to correlate with broker logs and Prometheus alerts.

## Metrics design

### Broker metrics

Prometheus scrapes these EMQX endpoints every 15 seconds:

- `http://mqtt:18083/api/v5/prometheus/stats?mode=node`
- `http://mqtt:18083/api/v5/prometheus/auth?mode=node`

These provide broker counters and gauges such as:

- current connections and sessions
- retained message count
- client connect/disconnect counters
- subscribe counters
- bytes in/out
- messages published, delivered, acknowledged, and dropped
- delivery-drop counters including queue-full conditions
- EMQX VM CPU and memory gauges

### Container and volume metrics

Native EMQX metrics do not cover everything required operationally, so the stack also scrapes:

- `cadvisor:8080` for container CPU, memory, and start-time signals
- `mqtt-volume-metrics:8000/metrics` for `./data/emqx` and `./logs/emqx` size/error gauges

### Cardinality posture

The monitoring config intentionally avoids per-topic labels and does not enable topic-level metrics in Prometheus. This keeps `market_id` and participant IDs from exploding cardinality when many simulations run in parallel.

## Alert rules included

The bundle ships initial rules for:

- `MQTTBrokerDown`
- `MQTTMetricsEndpointDown`
- `MQTTConnectionCountTooLow`
- `MQTTReconnectStorm`
- `MQTTDroppedMessagesDetected`
- `MQTTQueueBacklogHigh`
- `MQTTPersistenceError`
- `MQTTBrokerDiskUsageHigh`
- `MQTTBrokerRestartingFrequently`

Thresholds are intentionally conservative starter values and should be tuned for your expected participant count and message rates.

## Important limitation on queue-depth observability

EMQX's Prometheus surface is strong on counters and current broker totals, but it does **not** provide a clean current per-client queue-depth gauge in the default broker pull endpoints.

Because of that, `MQTTQueueBacklogHigh` uses `emqx_delivery_dropped_queue_full` as the first reliable alert signal. When it fires, inspect per-client pressure with:

```bash
docker exec mqtt /opt/emqx/bin/emqx ctl clients show <clientid>
docker exec mqtt /opt/emqx/bin/emqx ctl clients stats /tmp/clients.csv
```

Those commands surface client-level fields such as inflight, awaiting-rel, enqueued, and dropped counts.

## Logs

The broker uses console logging so Docker can collect logs from stdout/stderr. Default EMQX console logging is warning-level unless you change it.

For temporary troubleshooting, raise log verbosity through configuration or Dashboard, then drop it back once the issue is resolved. Avoid logging full message payloads by default.

## Resource policy

Starter limits are exposed in `.env.example`:

- broker CPU and memory
- Prometheus CPU and memory
- cAdvisor CPU and memory
- directory-metrics helper CPU and memory
- file-descriptor limits for the broker

The defaults are conservative and intended for local development and early integration.

## Security posture

### Baseline

- No MQTT auth or TLS is required initially.
- The broker is only bound to `127.0.0.1` on the host by default.
- The container image already runs as a non-root `emqx` user.
- `no-new-privileges` and `cap_drop: [ALL]` are applied to the broker container.

### Dashboard password

Set `EMQX_DASHBOARD_DEFAULT_PASSWORD` before the **first** boot. This only seeds the initial admin password; changing it later does not overwrite an already-initialized volume.

### Future auth/TLS path

When TREX is ready for credentials or TLS, extend the stack in this order:

1. Enable `listeners.ssl.default` in `emqx/base.hocon` and mount certs under `./emqx/certs`.
2. Add EMQX authentication (username/password or other supported authenticator).
3. Move secrets into Docker secrets or your deployment platform's secret store.
4. If you enable Basic Auth for the Prometheus pull endpoints, create an EMQX API key and add `basic_auth` to the Prometheus jobs.

## Acceptance checklist

### Connectivity

- [ ] TREX containers resolve `mqtt`
- [ ] TREX market, sim-controller, and participant processes connect with only config changes

### Protocol

- [ ] MQTT 5 user properties survive end to end
- [ ] QoS 1 publishes and subscriptions complete correctly
- [ ] QoS 2 publishes and subscriptions complete correctly
- [ ] Retained `join_market` messages are present for active participants
- [ ] Empty retained tombstones clear those retained presence topics on disconnect

### Simulation

- [ ] A full TREX simulation can start and complete without stalling
- [ ] Participants join and receive `market_info`
- [ ] Meter, `end_turn`, `end_round`, and settlement-ack flows complete reliably

### Monitoring

- [ ] Prometheus targets are up
- [ ] Broker connection and message-flow metrics are visible
- [ ] At least one alert can be forced and observed

### Restart

- [ ] Restarting `mqtt` yields predictable TREX reconnection
- [ ] Retained-state behavior after restart matches whether `./data/emqx` is preserved

## Useful operator commands

```bash
docker compose ps
docker compose logs -f mqtt
docker exec mqtt /opt/emqx/bin/emqx ctl status
docker exec mqtt /opt/emqx/bin/emqx ctl broker stats
docker exec mqtt /opt/emqx/bin/emqx ctl broker metrics
docker exec mqtt /opt/emqx/bin/emqx ctl clients list
docker exec mqtt /opt/emqx/bin/emqx ctl clients show <clientid>
docker exec mqtt /opt/emqx/bin/emqx ctl clients stats /tmp/clients.csv
```

## Notes for later managed deployment

This layout moves cleanly to a more managed environment:

- keep `mqtt` as the stable service DNS name or alias
- keep a persistent data volume for EMQX
- keep Prometheus scraping the EMQX pull endpoints
- replace cAdvisor and the directory-size helper with your platform's container/filesystem metrics if those are already available
