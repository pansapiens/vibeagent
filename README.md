# vibeagent

A terminal friendly LLM chat interface.

Features:

- MCP tool support
- Multiple OpenAI-compatible API endpoints (openrouter, and your local ollama/llama-swap)
- Context management
- Monitoring and telemetry via arize-phoenix, in-app context dump

> NOTE WELL: The default configuration has access to your filesystem and has the ability to cause data loss and maybe worse. You have been warned.

##  Quickstart

If you have [uv](https://docs.astral.sh/uv/) installed:

```bash
uvx --from git+https://github.com/pansapiens/vibeagent vibeagent
```

## Installing

If you've like to install it so you have the `vibeagent` command available:

```bash
pip install -U git+https://github.com/pansapiens/vibeagent

# Then run:
vibeagent
```

### Commands

- `/tools` - List available tools
- `/quit` - Quit the agent
- `/model [name]` - Change the model. If `[name]` is omitted, a selection dialog is shown.
- `/refresh-models` - Fetches the latest list of available models from configured endpoints.
- `/compress [strategy]` - Manually compress the conversation history to save tokens. Strategies: `drop_oldest`, `middle_out`, `summarize`.
- `/dump-context [format]` - Display the agent's current memory. Format can be `markdown` (default) or `json`.
- `/show-settings` - Displays the location of configuration files and their current content.

### Shell Commands

Commands starting with `!` are executed as shell commands in a pseudo-persistent shell session:

- `!ls` - List files in the current directory
- `!cd /path/to/directory` - Change directory
- `!pwd` - Show current working directory
- `!exit` - Exit the shell session (resets working directory and environment)

The `!` shell session starts in the first path from `allowedPaths` in settings (default: `$HOME/ai_workspace`). The working directory, environment variables, shell functions, aliases, and shell options persist between commands.

**Container Sandboxing**: When `containers.sandboxShell` is enabled in settings, shell commands run inside the configured container with `allowedPaths` mounted as volumes for isolation and security.

#### Redirect Commands

After executing a shell command, you can redirect its output to the LLM using these special commands:

- `!>` - Redirect both stdout and stderr from the last command to the LLM
- `!1>` - Redirect only stdout from the last command to the LLM
- `!2>` - Redirect only stderr from the last command to the LLM

You can also include a message with the redirect:

```
!> Please analyze this output
!1> Check what files were found
!2> Look at the error messages
```

The LLM will receive a formatted message including:
- Your optional message (if provided)
- The command that was executed
- The relevant output (stdout/stderr based on the redirect type)
- The exit code

Example output sent to LLM:
```
Please analyze this output

$ ls -la

STDOUT:

total 8
drwxr-xr-x 2 user user 4096 Jan 1 12:00 .
drwxr-xr-x 3 user user 4096 Jan 1 12:00 ..

STDERR:

ls: cannot access 'nonexistent': No such file or directory

EXITCODE:
1
```

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

### File Locations

`vibeagent` stores its files in standard user locations on your operating system:

- **Configuration**: `settings.json` is located in:
  - **Linux**: `~/.config/vibeagent/settings.json`
  - **macOS**: `~/Library/Application Support/vibeagent/settings.json` ?
  - **Windows**: `C:\Users\<user>\AppData\Local\vibeagent\vibeagent\settings.json` ?

- **Logs**: Log files are stored in:
  - **Linux**: `~/.local/state/vibeagent/log/`
  - **macOS**: `~/Library/Logs/vibeagent/` ?
  - **Windows**: `C:\Users\<user>\AppData\Local\vibeagent\vibeagent\Logs\` ?

### Configuration file

The chat agent can be configured in multiple ways with the following priority (highest to lowest):

1. **Command line arguments** (eg `--model`, `--api-key-env-var`, `--api-base`) - Overrides all other settings
2. **settings.json** - Model configuration in the settings file
3. **Environment variables** - Fallback to .env file

#### settings.json configuration files

```json
{
  "endpoints": {
    "openrouter": {
      "api_key": "$OPENROUTER_API_KEY",
      "api_base": "https://openrouter.ai/api/v1",
      "enabled": true
    },
    "ollama": {
      "api_key": "ollama",
      "api_base": "http://localhost:11434/v1",
      "enabled": false
    }
  },
  "defaultModel": "mistralai/devstral-small:free",
  "favoriteModels": [
    "mistralai/devstral-small:free",
    "google/gemma-3n-e4b-it"
  ],
  "allowCurrentDirectory": true,
  "allowedPaths": [
    "$HOME/ai_workspace"
  ],
  "containers": {
      "enabled": false,
      "engine": "docker",
      "image": "ghcr.io/pansapiens/vibeagent-mcp:latest",
      "home_mount_point": "/home/agent",
      "sandboxShell": false
  }
}
```

- `endpoints`: A dictionary of API providers. The agent will fetch available models from all `enabled` endpoints.
  - `api_key`: Your API key for the service (can use environment variable substitution with `$VAR_NAME`).
  - `api_base`: The API base URL for the service.
  - `enabled`: Set to `true` to use this endpoint. This is optional and defaults to `true`.
- `defaultModel`: The model identifier to use on startup.
- `favoriteModels`: A list of model IDs to show at the top of the `/model` selection list.
- `allowCurrentDirectory`: When `true` (default), automatically adds the directory where vibeagent is started to the beginning of `allowedPaths`. Set to `false` to disable this behavior.
- `allowedPaths`: A list of directories that vibeagent can access. The shell session starts in the first path in this list. Environment variables are supported (e.g., `$HOME/ai_workspace`).
- `containers`: Run MCP servers inside a container. `engine` can be `docker` or `apptainer`. `image` is the image to use. `sandboxShell` is whether to sandbox the `!` shell commands inside the container (requires `enabled: true`).

## Docker and apptainer container wrappers

Typical MCP servers can be run in a container to help restrict their access to the host filesystem. This is only a weak form of sandboxing, but better than nothing.

A Dockerfile `mcp.Dockerfile` is provided to build a container image that is typically compatible with most MCP servers.

### Using the pre-built image

The container image is automatically built and published to the GitHub Package Registry. You can pull it directly:

```bash
docker pull ghcr.io/pansapiens/vibeagent-mcp:latest
```

### Building locally

To build the image locally:

```bash
docker build -t vibeagent-mcp:latest -f mcp.Dockerfile .
```

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