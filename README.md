# vibeagent

##  Running

```bash
uv run vibeagent.py
```

Ask the agent to do something filesystem or search related.

### Commands

- `/tools` - List available tools
- `/quit` - Quit the agent
- `/model [name]` - Change the model. If `[name]` is omitted, a selection dialog is shown.

### Command Line Options

- `--model MODEL` - Override the model to use (overrides settings.json and .env)
- `--api-key-env-var VAR` - Environment variable name to get API key from (overrides settings.json)
- `--api-base URL` - Override the API base URL to use (overrides settings.json and .env)

Examples:
```bash
uv run vibeagent.py --model "google/gemma-3n-e4b-it"
uv run vibeagent.py --model "anthropic/claude-3.5-sonnet" --api-key-env-var "ANTHROPIC_API_KEY"
uv run vibeagent.py --api-base "http://127.0.0.1:11434" --model "qwen3:8b" --api-key-env-var "OLLAMA_API_KEY"
```

## Configuring

```bash
cp dot.env.example .env
```
Edit `.env` to your liking.

After first run, a `settings.json` will be generated. You can edit this to add or remove MCP servers and configure the model.

### Model Configuration

The model can be configured in multiple ways with the following priority (highest to lowest):

1. **Command line arguments** (`--model`, `--api-key-env-var`, `--api-base`) - Overrides all other settings
2. **settings.json** - Model configuration in the settings file
3. **Environment variables** - Fallback to .env file

#### settings.json Model Configuration

```json
{
  "model": {
    "id": "mistralai/devstral-small:free",
    "api_key": "$OPENROUTER_API_KEY",
    "api_base": "https://openrouter.ai/api/v1"
  },
  "favoriteModels": [
    "mistralai/devstral-small:free",
    "google/gemma-3n-e4b-it"
  ]
}
```

- `id`: The model identifier (e.g., "mistralai/devstral-small:free", "google/gemma-3n-e4b-it")
- `api_key`: Your API key (can use environment variable substitution with `$VAR_NAME`)
- `api_base`: The API base URL (defaults to OpenRouter)
- `favoriteModels`: A list of model IDs to show at the top of the model selection list.

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