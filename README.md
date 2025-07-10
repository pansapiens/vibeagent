# vibeagent

##  Running

```bash
uv run vibeagent.py
```

Ask the agent to do something filesystem or search related.

### Commands

- `/tools` - List available tools
- `/quit` - Quit the agent

## Configuring

```bash
cp dot.env.example .env
```
Edit `.env` to your liking.

After first run, a `settings.json` will be generated. You can edit this to add or remove MCP servers.

##  Monitoring the agent, telemetry

Telemetry is automatically enabled when Phoenix is running on the configured endpoint. The telemetry is non-blocking, so the application will continue to work even if Phoenix goes down mid-session.

### Configuration

Add these to your `.env` file:

```bash
# Telemetry endpoint (default: http://localhost:4317)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Disable telemetry completely
DISABLE_TELEMETRY=false
```

### Setup

To enable telemetry monitoring:

```bash
uv venv
source .venv/bin/activate
uv pip install arize-phoenix
phoenix serve
```