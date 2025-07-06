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

```bash
uv venv
source .venv/bin/activate
uv pip install arize-phoenix
phoenix serve
```