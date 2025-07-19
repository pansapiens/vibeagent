# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-07-19

### Added
- **GitHub Actions**: Automated Docker image builds and publishing to GitHub Package Registry
- **Container image management**: Automatic checking and pulling of Docker/Apptainer images before starting MCP servers
- **Session management**: `/save`, `/load`, `/delete` commands for managing chat sessions
- **Auto-save functionality**: Session transcripts are now automatically saved after each agent response when `"autoSave": true` is set in settings.json
- **Shell command integration**: `!` commands to run shell commands and `!>` to send last shell output to LLM
- **Docker/Apptainer support**: MCP servers can now run in containerized environments
- **Slash command autocomplete**: Tab completion for all slash commands
- **Context management**: `/compress` and `/dump-context` commands for managing conversation context
- **Model selection**: In-app model selection with `/refresh-models` command
- **MCP server management**: `/tools` command to list available tools and their associated MCP servers
- **Keyboard shortcuts**: Esc key to cancel ongoing requests
- **Multiple provider support**: Configuration now supports multiple LLM providers
- **Platform-specific paths**: Settings and logs now use platform-appropriate directories
- **Async initialization**: UI shows immediately while MCP servers start up in background
- **Pip installation**: Project is now installable via pip with pyproject.toml
- **Default MCP servers**: Added text-editor, pollinations, and searxng MCP servers to defaults
- **Model display**: Model ID shown in footer and initial chat messages
- **Input focus**: Input widget automatically focused on startup
- **Error handling**: Improved error handling for agent responses and UI updates
- **Instrumentation**: Configurable, non-blocking telemetry support
- **Command line arguments**: Support for `--model` and other arguments
- **Settings management**: `/show-settings` command to locate configuration files
- **Cyberpunk theme**: Modern dark theme with improved loading indicators
- **License**: Added MIT license file

### Changed
- **Configuration format**: Changed from single provider to multiple provider support
- **MCP server settings**: Changed from "disabled" to "enabled" field for MCP server configuration
- **Settings location**: Moved from local directory to platform-specific paths
- **UI improvements**: Better loading indicators and internal CSS handling
- **Session saving**: Auto-save sessions are named with timestamp format: `_autosave_YYYYMMDD_HHMMSS.json`
- **Auto-save errors**: Logged but don't display UI messages to avoid cluttering the interface
- **Context compression**: Improved summarization compression algorithm
- **Error messages**: Refactored to display in UI and logs consistently

### Fixed
- **Redundant messages**: Fixed duplicate UI messages for `!>` commands
- **MCP server disabling**: Fixed issue with disabling MCP servers in settings.json
- **Agent response errors**: Improved error handling to ensure UI updates properly
- **Aider tool**: Fixed integration with aider tool
- **Endpoint defaults**: Endpoint enabled setting defaults to true if missing

### Technical
- **Dependencies**: Added textual-autocomplete, platformdirs, and other required packages
- **Project structure**: Added proper Python packaging with pyproject.toml
- **Logging**: Improved log file path and model name logging
- **Async operations**: Made agent initialization non-blocking
- **Container support**: Added mcp.Dockerfile for containerized MCP servers
- **CI/CD**: Added GitHub Actions workflow for automated Docker image builds 