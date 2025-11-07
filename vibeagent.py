#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "smolagents[litellm,mcp,telemetry,toolkit]",
#   "rich",
#   "prompt-toolkit",
#   "python-dotenv",
#   "aider-chat",
#   "openai",
#   "platformdirs",
#   "tiktoken",
# ]
# ///

import logging
import sys
import json
import os
import io
import argparse
import time
import random
import string
import shlex
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from dotenv import load_dotenv


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.encoding = "utf-8"

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.get_logger().log(self.level, line.rstrip())

    def flush(self):
        pass

# MCP imports
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, tool, MCPClient
from contextlib import redirect_stdout
import platformdirs
import tiktoken
from smolagents.models import OpenAIServerModel, MessageRole, ChatMessage, TokenUsage
from smolagents.monitoring import Timing
from mcp import StdioServerParameters

# Rich imports
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from rich.prompt import Prompt, Confirm
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import PromptSession

# Smolagents imports
from smolagents.memory import ActionStep, MemoryStep, TaskStep
import uuid

try:
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry import trace
    from phoenix.otel import register

    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


def get_logger():
    """Get a properly configured get_logger()."""
    return logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration."""

    def __init__(self):
        # Use separate directories for config and data
        self.config_dir = Path(platformdirs.user_config_dir("vibeagent", "pansapiens"))
        self.data_dir = Path(platformdirs.user_data_dir("vibeagent", "pansapiens"))

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # For backward compatibility
        self.app_dir = self.data_dir

        self.settings_file = self.config_dir / "settings.json"
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.load_settings()

    def load_settings(self):
        """Load settings from file or create defaults."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            except Exception as e:
                get_logger().error(f"Error loading settings: {e}")
                self.settings = self._default_settings()
        else:
            self.settings = self._default_settings()
            self.save_settings()

    def _default_settings(self):
        """Create default settings."""
        return {
            "model": os.getenv("VIBEAGENT_MODEL", "openai:gpt-4"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "api_base": os.getenv("VIBEAGENT_API_BASE", None),
            "allowedPaths": [str(Path.cwd())],
            "mcpServers": {},
            "max_tokens": 4096,
            "temperature": 0.7,
            "session_id": None
        }

    def save_settings(self):
        """Save settings to file."""
        try:
            temp_file = self.settings_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            temp_file.replace(self.settings_file)
        except Exception as e:
            get_logger().error(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get setting value."""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set setting value."""
        self.settings[key] = value
        self.save_settings()


class MessageRenderer:
    """Handles rendering of different message types."""

    def __init__(self, console: Console):
        self.console = console

    def render_message(self, message_type: str, content: str, title: str = None) -> Panel:
        """Render different message types with appropriate styling."""

        styles = {
            "user": {
                "title": title or "[bold blue]User[/bold blue]",
                "border_style": "blue",
                "title_align": "left"
            },
            "agent": {
                "title": title or "[bold magenta]Agent[/bold magenta]",
                "border_style": "magenta",
                "title_align": "left"
            },
            "tool": {
                "title": title or "[bold green]Tool[/bold green]",
                "border_style": "green",
                "title_align": "left"
            },
            "error": {
                "title": title or "[bold red]Error[/bold red]",
                "border_style": "red",
                "title_align": "left"
            },
            "info": {
                "title": title or "[bold cyan]Info[/bold cyan]",
                "border_style": "cyan",
                "title_align": "left"
            },
            "system": {
                "title": title or "[bold yellow]System[/bold yellow]",
                "border_style": "yellow",
                "title_align": "left"
            }
        }

        style = styles.get(message_type, styles["info"])

        # Format content based on type
        if message_type == "agent":
            try:
                formatted_content = Markdown(content)
            except Exception:
                formatted_content = Text(content)
        elif message_type == "tool" and "```" in content:
            # Try to detect and syntax highlight code blocks
            formatted_content = Markdown(content)
        elif message_type == "error":
            formatted_content = Text(content, style="red")
        else:
            formatted_content = Text(content)

        return Panel(
            formatted_content,
            title=style["title"],
            border_style=style["border_style"],
            title_align=style["title_align"],
            padding=(0, 1),
            expand=True,
            box=box.HORIZONTALS
        )


class InputHandler:
    """Manages user input with history and completion."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.history_file = config_manager.app_dir / "input_history.txt"
        self.key_bindings = KeyBindings()
        self.setup_key_bindings()

        # Command completion with all available commands
        self.base_commands = [
            "/help", "/quit", "/exit", "/clear",
            "/save", "/load", "/delete", "/tools",
            "/model", "/refresh-models", "/compress",
            "/dump-context", "/show-settings", "/status"
        ]

        # Dynamic completion that will be updated based on context
        self.completer = WordCompleter(self.base_commands, ignore_case=True)

        self.session = PromptSession(
            history=FileHistory(str(self.history_file)),
            completer=self.completer,
            key_bindings=self.key_bindings,
            multiline=False,
            wrap_lines=True
        )

    def setup_key_bindings(self):
        """Setup custom key bindings."""
        @self.key_bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C to exit."""
            event.app.exit()

    def get_dynamic_completions(self, text: str):
        """Get dynamic completions based on current context."""
        if not text.startswith("/"):
            return []

        parts = text.split(" ", 1)
        command = parts[0].lower()

        if len(parts) == 1:
            # User is typing a command
            return self.base_commands
        else:
            # User has typed a command and is typing arguments
            arg = parts[1] if len(parts) > 1 else ""

            if command == "/save" or command == "/load" or command == "/delete":
                # Suggest session names
                sessions = self._get_session_names()
                return [f"{command} {session}" for session in sessions]
            elif command == "/model":
                # Suggest common models
                models = [
                    "openai:gpt-4o",
                    "openai:gpt-4o-mini",
                    "openai:gpt-3.5-turbo",
                    "anthropic:claude-3-5-sonnet-20241022",
                    "ollama:llama3.1"
                ]
                return [f"/model {model}" for model in models]
            elif command == "/compress":
                strategies = ["drop_oldest", "middle_out", "summarize"]
                return [f"/compress {strategy}" for strategy in strategies]
            elif command == "/dump-context":
                formats = ["markdown", "json"]
                return [f"/dump-context {fmt}" for fmt in formats]

        return []

    def _get_session_names(self):
        """Get list of available session names."""
        sessions = []
        # Get custom session names
        custom_sessions_dir = self.config_manager.app_dir / "custom_sessions"
        if custom_sessions_dir.exists():
            for session_file in custom_sessions_dir.glob("*.json"):
                sessions.append(session_file.stem)
        return sessions

    def get_input(self, prompt_text: str = "> ") -> str:
        """Get user input with prompt-toolkit and autocompletion."""
        try:
            user_input = self.session.prompt(prompt_text)
            return user_input.strip()
        except KeyboardInterrupt:
            return "/quit"
        except EOFError:
            return "/quit"


class AgentManager:
    """Manages agent initialization and communication."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agent = None
        self.mcp_clients = {}  # Store MCPClient objects
        self.mcp_tools = []   # Store collected MCP tools
        self.agent_initialized = False
        self.agent_error = None
        self.model_details = {}
        self.available_models = []

        # Initialize built-in tools
        self.built_in_tools = self._create_built_in_tools()

        # Initialize MCP servers first
        self.initialize_mcp_servers()

        # Initialize agent (but don't block on API calls)
        self.initialize_agent()

    def initialize_agent(self):
        """Initialize the smolagents agent with proper configuration."""
        try:
            get_logger().info("Initializing agent with configuration...")

            # Get model configuration
            model_config = self.config_manager.settings
            endpoints = model_config.get("endpoints", {})
            default_model = model_config.get("defaultModel", "openai:gpt-4")

            get_logger().info(f"Default model: {default_model}")

            # Create fallback model details for configured models
            self._create_fallback_model_details(endpoints, default_model)

            # Initialize agent with default model if available
            if default_model in self.model_details:
                self._create_agent_for_model(default_model)
                get_logger().info(f"Agent initialized with model: {default_model}")
            else:
                get_logger().warning(f"Model details not found for {default_model}, agent will be initialized on demand")

            self.agent_initialized = True
            return True

        except Exception as e:
            get_logger().error(f"Error initializing agent: {e}")
            self.agent_error = str(e)
            self.agent = None
            self.agent_initialized = True
            return False

    def _create_fallback_model_details(self, endpoints: dict, default_model: str):
        """Create fallback model details for configured models."""
        if not endpoints:
            get_logger().warning("No endpoints configured")
            return

        # Find default provider (marked as default or first enabled)
        default_provider = None
        for name, config in endpoints.items():
            if config.get("enabled", True) and config.get("default"):
                default_provider = name
                break

        if not default_provider:
            for name, config in endpoints.items():
                if config.get("enabled", True):
                    default_provider = name
                    break

        if not default_provider:
            get_logger().warning("No enabled endpoints found")
            return

        default_config = endpoints[default_provider]
        api_key = self._substitute_env_vars(default_config.get("api_key", ""))
        api_base = default_config.get("api_base", "https://api.openai.com/v1")

        # Create fallback details for default model
        if default_model not in self.model_details:
            self.model_details[default_model] = {
                "provider": default_provider,
                "original_id": default_model,
                "api_key": api_key,
                "api_base": api_base,
            }
            get_logger().info(f"Created fallback model details for {default_model} using provider {default_provider}")

    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in configuration values."""
        if isinstance(value, str) and value.startswith("$") and value[1:] in os.environ:
            return os.environ[value[1:]]
        return value

    def _create_built_in_tools(self):
        """Create built-in tools for the agent."""
        @tool
        def aider_edit_file(file: str, instruction: str) -> str:
            """
            Edits a file using the aider tool with a given instruction.
            Args:
                file: The path to the file to edit.
                instruction: The instruction message to pass to aider for the edit.
            """
            output_capture = io.StringIO()
            try:
                # Get current model details
                current_model = self.config_manager.get("defaultModel", "openai:gpt-4")
                model_details = self.model_details.get(current_model)
                if not model_details:
                    return f"Error: Could not find model details for {current_model}. Cannot run aider."

                with redirect_stdout(output_capture):
                    # Import aider dependencies here to avoid circular dependencies
                    from aider.coders import Coder
                    from aider.models import Model

                    # Setup model and coder
                    model = Model(model_details["original_id"])
                    coder = Coder.create(
                        main_model=model,
                        fnames=[file] if file else None,
                        show_diffs=False,
                        auto_commits=False,
                        pretty_output=False,
                    )

                    # Run the edit instruction
                    coder.run(instruction)

                captured_output = output_capture.getvalue()
                return f"Aider edit completed:\n{captured_output}"

            except Exception as e:
                return f"Error running aider: {str(e)}\nCaptured output:\n{output_capture.getvalue()}"

        return [aider_edit_file]

    def initialize_mcp_servers(self):
        """Initialize MCP servers and collect their tools."""
        try:
            get_logger().info("Initializing MCP servers...")

            server_configs = self.config_manager.settings.get("mcpServers", {})
            get_logger().info(f"Found {len(server_configs)} MCP server configurations.")

            if not server_configs:
                get_logger().info("No MCP servers configured")
                return

            for name, config in server_configs.items():
                get_logger().info(f"Processing MCP server config: {name}")

                # Skip disabled servers
                if not config.get("enabled", True):
                    get_logger().info(f"Skipping disabled MCP server: {name}")
                    continue

                command = config.get("command")
                args = config.get("args", [])
                env = config.get("environment", config.get("env", {}))

                if not command:
                    get_logger().warning(f"No command specified for MCP server: {name}")
                    continue

                # Prepare environment
                full_env = {**os.environ, **env}

                try:
                    # Create server parameters
                    server_param = StdioServerParameters(
                        command=command,
                        args=args,
                        env=full_env,
                    )

                    get_logger().info(f"[{name}] Creating MCPClient...")
                    client = MCPClient([server_param])

                    # Store the client
                    self.mcp_clients[name] = client

                    get_logger().info(f"[{name}] Getting tools from server...")
                    server_tools = client.get_tools()
                    get_logger().info(f"[{name}] Found {len(server_tools)} tools.")

                    # Add to our tools list
                    self.mcp_tools.extend(server_tools)

                except Exception as e:
                    get_logger().error(f"[{name}] Failed to initialize MCP server: {e}")
                    continue

            get_logger().info(f"MCP initialization complete. Collected {len(self.mcp_tools)} tools from {len(self.mcp_clients)} servers.")

        except Exception as e:
            get_logger().error(f"Error during MCP server initialization: {e}")

    def _create_agent_for_model(self, model_id: str):
        """Create agent for specific model."""
        if model_id not in self.model_details:
            raise Exception(f"Model details not found for {model_id}")

        details = self.model_details[model_id]

        # Create OpenAI Server model
        model = OpenAIServerModel(
            model_id=details["original_id"],
            api_key=details["api_key"],
            api_base=details["api_base"],
            max_tokens=self.config_manager.get("max_tokens", 4096),
            temperature=self.config_manager.get("temperature", 0.7)
        )

        # Collect all tools: built-in + MCP
        all_tools = []

        # Add built-in tools
        if hasattr(self, 'built_in_tools') and self.built_in_tools:
            all_tools.extend(self.built_in_tools)
            get_logger().info(f"Added {len(self.built_in_tools)} built-in tools")

        # Add MCP tools
        if hasattr(self, 'mcp_tools') and self.mcp_tools:
            all_tools.extend(self.mcp_tools)
            get_logger().info(f"Added {len(self.mcp_tools)} MCP tools from {len(self.mcp_clients)} servers")

        # Create agent with all tools
        self.agent = ToolCallingAgent(
            model=model,
            tools=all_tools,
            max_steps=10,
            verbosity_level=1
        )

        get_logger().info(f"Created agent for model: {model_id} with {len(all_tools)} total tools")

    def run_agent(self, user_message: str, callback=None):
        """Run agent with user message."""
        if not self.agent:
            # Try to initialize with default model
            default_model = self.config_manager.get("defaultModel", "openai:gpt-4")
            if default_model in self.model_details:
                self._create_agent_for_model(default_model)
                get_logger().info(f"Agent initialized on-demand with model: {default_model}")
            else:
                raise Exception("Agent not available - please check your configuration")

        try:
            # Intercept smolagents monitoring output to replace "New run" with "Sending..."
            # Capture stdout to intercept "New run" output
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                response = self.agent.run(user_message)

            # Process captured output and replace "New run" with "Sending..."
            captured_text = captured_output.getvalue()
            if captured_text.strip():
                # Replace "New run" with "Sending..." in the captured output
                modified_text = captured_text.replace("New run", "Sending...")
                # Print the modified text to our console
                print(modified_text, end='', flush=True)

            if callback:
                callback(response)

            return response

        except Exception as e:
            get_logger().error(f"Error running agent: {e}")
            raise


class SessionManager:
    """Handles session persistence and restoration."""

    def __init__(self, config_manager: ConfigManager, agent_manager: AgentManager):
        self.config_manager = config_manager
        self.agent_manager = agent_manager
        self.session_id = None
        self.chat_history = []

    def new_session(self):
        """Create a new session."""
        self.session_id = str(uuid.uuid4())
        self.chat_history = []
        self.config_manager.set("session_id", self.session_id)
        get_logger().info(f"Created new session: {self.session_id}")

    def save_session(self):
        """Save current session to file."""
        if not self.session_id:
            return

        session_file = self.config_manager.sessions_dir / f"{self.session_id}.json"

        try:
            session_data = {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
                "chat_history": self.chat_history,
                "settings": self.config_manager.settings
            }

            temp_file = session_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            temp_file.replace(session_file)

            get_logger().info(f"Session saved: {session_file}")

        except Exception as e:
            get_logger().error(f"Error saving session: {e}")

    def load_session(self, session_id: str):
        """Load session from file."""
        session_file = self.config_manager.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            get_logger().error(f"Session file not found: {session_file}")
            return False

        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            self.session_id = session_data["session_id"]
            self.chat_history = session_data.get("chat_history", [])

            # Restore agent memory if available
            if self.agent_manager.agent and hasattr(self.agent_manager.agent, 'memory'):
                # TODO: Implement memory restoration
                pass

            get_logger().info(f"Session loaded: {self.session_id}")
            return True

        except Exception as e:
            get_logger().error(f"Error loading session: {e}")
            return False

    def add_message(self, message_type: str, content: str, title: str = None):
        """Add a message to chat history."""
        message = {
            "type": message_type,
            "content": content,
            "title": title,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_history.append(message)

    def list_sessions(self):
        """List all available sessions."""
        sessions = []
        for session_file in self.config_manager.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                sessions.append({
                    "id": session_data["session_id"],
                    "created_at": session_data["created_at"],
                    "message_count": len(session_data.get("chat_history", []))
                })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


class RichChatApp:
    """Main application using Rich for display."""

    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager()
        self.message_renderer = MessageRenderer(self.console)
        self.input_handler = InputHandler(self.config_manager)
        self.agent_manager = AgentManager(self.config_manager)
        self.session_manager = SessionManager(self.config_manager, self.agent_manager)

        self.running = False
        self.live = None

        # Load or create session
        session_id = self.config_manager.get("session_id")
        if session_id and self.session_manager.load_session(session_id):
            pass  # Session loaded successfully
        else:
            self.session_manager.new_session()

    def render_chat_history(self):
        """Render the complete chat history."""
        if not self.session_manager.chat_history:
            return Text("No messages yet. Start a conversation!", style="dim")

        from rich.console import Group
        from rich.align import Align

        messages = []
        for msg in self.session_manager.chat_history:
            panel = self.message_renderer.render_message(
                msg["type"],
                msg["content"],
                msg.get("title")
            )
            messages.append(panel)
            messages.append("")  # Add spacing

        return Group(*messages)

    def display_welcome(self):
        """Display welcome message."""
        welcome_text = Text()
        welcome_text.append("🤖 ", style="bold magenta")
        welcome_text.append("vibeagent", style="bold cyan")
        welcome_text.append(" - Rich Chat Interface\n\n", style="bold")

        if self.agent_manager.agent:
            welcome_text.append("✅ Agent connected and ready!\n", style="green")
        else:
            welcome_text.append("⚠️  Agent not initialized (will be initialized on demand)\n", style="yellow")

        welcome_text.append("\nKey Commands:\n", style="bold")
        welcome_text.append("  /help  - Show all available commands\n", style="cyan")
        welcome_text.append("  /tools - List available tools\n", style="cyan")
        welcome_text.append("  /model - Change or show current model\n", style="cyan")
        welcome_text.append("  /save [name] - Save session\n", style="cyan")
        welcome_text.append("  /load [name] - Load session\n", style="cyan")
        welcome_text.append("  !command - Run shell commands\n", style="yellow")
        welcome_text.append("  /quit  - Exit application\n", style="cyan")
        welcome_text.append("\nType your message and press Enter to chat.\n", style="dim")

        self.console.print(Panel(welcome_text, border_style="cyan"))

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        if command.startswith('/'):
            parts = command.strip().split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ['/quit', '/exit']:
                self.running = False
                return True
            elif cmd == '/help':
                self.display_help()
                return True
            elif cmd == '/clear':
                self.session_manager.chat_history.clear()
                self.console.print("[green]Chat history cleared.[/green]")
                return True
            elif cmd == '/save':
                self.handle_save_command(arg)
                return True
            elif cmd == '/load':
                self.handle_load_command(arg)
                return True
            elif cmd == '/delete':
                self.handle_delete_command(arg)
                return True
            elif cmd == '/tools':
                self.list_tools()
                return True
            elif cmd == '/model':
                self.handle_model_command(arg)
                return True
            elif cmd == '/refresh-models':
                self.refresh_models()
                return True
            elif cmd == '/compress':
                self.compress_context(arg)
                return True
            elif cmd == '/dump-context':
                self.dump_context(arg)
                return True
            elif cmd == '/show-settings':
                self.show_settings()
                return True
            elif cmd == '/status':
                self.display_status()
                return True

        return False

    def display_help(self):
        """Display help information."""
        help_text = Text()
        help_text.append("Available Commands:\n\n", style="bold")

        commands = [
            ("/help", "Show this help message"),
            ("/quit, /exit", "Exit the application"),
            ("/clear", "Clear the chat history"),
            ("/save [name]", "Save current session (optional name)"),
            ("/load [name]", "Load a session (shows previous if no name)"),
            ("/delete <name>", "Delete a saved session"),
            ("/tools", "List available tools from MCP servers"),
            ("/model [model]", "Change model or show current model"),
            ("/refresh-models", "Refresh model list from API"),
            ("/compress [strategy]", "Compress context (drop_oldest, middle_out, summarize)"),
            ("/dump-context [format]", "Show agent's current memory (markdown/json)"),
            ("/show-settings", "Show configuration file locations and content"),
            ("/status", "Show current status and configuration"),
        ]

        for cmd, desc in commands:
            help_text.append(f"  {cmd:<25} - {desc}\n", style="cyan")

        help_text.append("\nShell Commands (start with !):", style="bold")
        help_text.append("  !ls -la              - List files in current directory\n", style="yellow")
        help_text.append("  !pwd                 - Show current working directory\n", style="yellow")
        help_text.append("  !echo 'hello'        - Run any shell command\n", style="yellow")
        help_text.append("\nAny other text will be sent to the agent.", style="dim")

        self.console.print(Panel(help_text, title="[bold]Help[/bold]", border_style="blue"))

    def display_status(self):
        """Display current status."""
        status_table = Table(title="System Status", show_header=True, header_style="bold magenta")
        status_table.add_column("Setting", style="cyan", width=20)
        status_table.add_column("Value", style="white")

        # Agent status
        agent_status = "✅ Connected" if self.agent_manager.agent else "❌ Not connected"
        status_table.add_row("Agent", agent_status)

        # Model info
        model = self.config_manager.get("defaultModel", "Unknown")
        status_table.add_row("Model", model)

        # Session info
        status_table.add_row("Session ID", self.session_manager.session_id[:8] + "...")
        status_table.add_row("Messages", str(len(self.session_manager.chat_history)))

        # Working directory
        status_table.add_row("Working Dir", os.getcwd())

        self.console.print(status_table)

    def handle_save_command(self, session_name: str = None):
        """Handle save command with optional session name."""
        if session_name:
            # Save with custom name
            custom_sessions_dir = self.config_manager.app_dir / "custom_sessions"
            custom_sessions_dir.mkdir(exist_ok=True)
            session_file = custom_sessions_dir / f"{session_name}.json"

            try:
                session_data = {
                    "session_id": self.session_manager.session_id,
                    "custom_name": session_name,
                    "created_at": datetime.now().isoformat(),
                    "chat_history": self.session_manager.chat_history,
                    "settings": self.config_manager.settings
                }

                temp_file = session_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                temp_file.replace(session_file)

                self.console.print(f"[green]Session saved as: {session_name}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error saving session: {e}[/red]")
        else:
            # Save with auto-generated name
            self.session_manager.save_session()
            self.console.print(f"[green]Session saved: {self.session_manager.session_id}[/green]")

    def handle_load_command(self, session_name: str = None):
        """Handle load command with optional session name."""
        if not session_name:
            # Load previous session if available
            sessions = self.session_manager.list_sessions()
            if sessions:
                previous_session = sessions[0]  # Most recent
                success = self.session_manager.load_session(previous_session["id"])
                if success:
                    self.console.print(f"[green]Loaded previous session: {previous_session['id'][:8]}...[/green]")
                else:
                    self.console.print("[red]Failed to load previous session[/red]")
            else:
                self.console.print("[yellow]No previous session found. Use '/load <name>' to load a specific session.[/yellow]")
        else:
            # Load specific named session
            custom_sessions_dir = self.config_manager.app_dir / "custom_sessions"
            session_file = custom_sessions_dir / f"{session_name}.json"

            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)

                    self.session_manager.chat_history = session_data.get("chat_history", [])
                    self.session_manager.session_id = session_data.get("session_id", str(uuid.uuid4()))

                    self.console.print(f"[green]Loaded session: {session_name}[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error loading session: {e}[/red]")
            else:
                self.console.print(f"[red]Session '{session_name}' not found[/red]")

    def handle_delete_command(self, session_name: str):
        """Handle delete command for named sessions."""
        if not session_name:
            self.console.print("[yellow]Usage: /delete <session_name>[/yellow]")
            return

        custom_sessions_dir = self.config_manager.app_dir / "custom_sessions"
        session_file = custom_sessions_dir / f"{session_name}.json"

        if session_file.exists():
            try:
                session_file.unlink()
                self.console.print(f"[green]Deleted session: {session_name}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error deleting session: {e}[/red]")
        else:
            self.console.print(f"[red]Session '{session_name}' not found[/red]")

    def list_tools(self):
        """List available tools from built-in and MCP servers."""
        tools_table = Table(title="Available Tools", show_header=True, header_style="bold green")
        tools_table.add_column("Tool Name", style="cyan")
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Source", style="magenta")

        # Show built-in tools
        if hasattr(self.agent_manager, 'built_in_tools') and self.agent_manager.built_in_tools:
            for tool in self.agent_manager.built_in_tools:
                tool_name = getattr(tool, 'name', str(tool))
                tool_desc = getattr(tool, 'description', 'No description')
                tools_table.add_row(tool_name, tool_desc, "Built-in")

        # Show MCP tools
        if hasattr(self.agent_manager, 'mcp_tools') and self.agent_manager.mcp_tools:
            for tool in self.agent_manager.mcp_tools:
                tool_name = getattr(tool, 'name', str(tool))
                tool_desc = getattr(tool, 'description', 'No description')
                tools_table.add_row(tool_name, tool_desc, "MCP")

        # Show agent's current tools (fallback)
        if (not hasattr(self.agent_manager, 'built_in_tools') or not self.agent_manager.built_in_tools) and \
           (not hasattr(self.agent_manager, 'mcp_tools') or not self.agent_manager.mcp_tools):
            if self.agent_manager.agent and hasattr(self.agent_manager.agent, 'tools') and self.agent_manager.agent.tools:
                for tool in self.agent_manager.agent.tools:
                    tool_name = getattr(tool, 'name', str(tool))
                    tool_desc = getattr(tool, 'description', 'No description')
                    tools_table.add_row(tool_name, tool_desc, "Agent")
            else:
                tools_table.add_row("No tools available", "Agent not initialized", "-")

        self.console.print(tools_table)

        # Show configured MCP servers
        mcp_servers = self.config_manager.settings.get("mcpServers", {})
        if mcp_servers:
            mcp_table = Table(title="MCP Servers Configuration", show_header=True, header_style="bold blue")
            mcp_table.add_column("Server", style="green")
            mcp_table.add_column("Status", style="white")
            mcp_table.add_column("Command", style="dim")

            for server_name, server_config in mcp_servers.items():
                enabled = server_config.get("enabled", False)
                status = "✅ Enabled" if enabled else "❌ Disabled"
                command = server_config.get("command", "N/A")
                mcp_table.add_row(server_name, status, command)

            self.console.print(mcp_table)
        else:
            self.console.print("[yellow]No MCP servers configured[/yellow]")

    def handle_model_command(self, model_name: str = None):
        """Handle model command to change or show current model."""
        current_model = self.config_manager.get("defaultModel", "Unknown")

        if model_name:
            # Change model
            model_name = model_name.strip()
            try:
                # Create fallback model details if needed
                endpoints = self.config_manager.settings.get("endpoints", {})
                if model_name not in self.agent_manager.model_details:
                    self.agent_manager._create_fallback_model_details(endpoints, model_name)

                if model_name in self.agent_manager.model_details:
                    self.agent_manager._create_agent_for_model(model_name)
                    self.config_manager.set("defaultModel", model_name)
                    self.config_manager.save_settings()
                    self.console.print(f"[green]Model changed to: {model_name}[/green]")
                else:
                    self.console.print(f"[red]Model '{model_name}' not found in configuration[/red]")
            except Exception as e:
                self.console.print(f"[red]Error changing model: {e}[/red]")
        else:
            # Show current model
            self.console.print(f"[cyan]Current model: {current_model}[/cyan]")

            # Show available models from configuration
            if self.agent_manager.model_details:
                self.console.print("[bold]Available models:[/bold]")
                for model_id in self.agent_manager.model_details.keys():
                    details = self.agent_manager.model_details[model_id]
                    provider = details.get("provider", "unknown")
                    current_marker = " (current)" if model_id == current_model else ""
                    self.console.print(f"  {model_id} - {provider}{current_marker}")
            else:
                self.console.print("[yellow]No models configured[/yellow]")

    def refresh_models(self):
        """Refresh model list from API (placeholder for now)."""
        self.console.print("[yellow]Model refresh not yet implemented in Rich version[/yellow]")
        self.console.print(f"[cyan]Current model: {self.config_manager.get('model', 'Unknown')}[/cyan]")

    def compress_context(self, strategy: str = None):
        """Compress conversation context."""
        if not strategy:
            strategy = "drop_oldest"  # Default strategy

        valid_strategies = ["drop_oldest", "middle_out", "summarize"]
        if strategy not in valid_strategies:
            self.console.print(f"[red]Invalid strategy. Valid options: {', '.join(valid_strategies)}[/red]")
            return

        # Simple implementation - remove oldest messages
        if strategy == "drop_oldest" and len(self.session_manager.chat_history) > 4:
            # Keep last 2 user messages and their responses
            messages_to_keep = self.session_manager.chat_history[-4:]
            removed_count = len(self.session_manager.chat_history) - 4
            self.session_manager.chat_history = messages_to_keep
            self.console.print(f"[green]Compressed context: removed {removed_count} old messages[/green]")
        elif strategy == "middle_out":
            # Remove some messages from the middle
            if len(self.session_manager.chat_history) > 6:
                # Keep first 2 and last 4 messages
                new_history = self.session_manager.chat_history[:2] + self.session_manager.chat_history[-4:]
                removed_count = len(self.session_manager.chat_history) - len(new_history)
                self.session_manager.chat_history = new_history
                self.console.print(f"[green]Compressed context: removed {removed_count} middle messages[/green]")
        elif strategy == "summarize":
            self.console.print("[yellow]Context summarization not yet implemented[/yellow]")
        else:
            self.console.print("[yellow]Not enough messages to compress[/yellow]")

    def dump_context(self, format_type: str = "markdown"):
        """Display agent's current memory/context."""
        if format_type not in ["markdown", "json"]:
            format_type = "markdown"

        if format_type == "json":
            context_data = {
                "session_id": self.session_manager.session_id,
                "model": self.config_manager.get("model"),
                "message_count": len(self.session_manager.chat_history),
                "chat_history": self.session_manager.chat_history,
                "settings": self.config_manager.settings
            }
            context_json = json.dumps(context_data, indent=2)
            self.console.print(Panel(context_json, title="[bold]Agent Context (JSON)[/bold]", border_style="blue"))
        else:
            # Markdown format
            context_md = f"""
# Agent Context

**Session ID:** {self.session_manager.session_id}
**Model:** {self.config_manager.get("model")}
**Messages:** {len(self.session_manager.chat_history)}

## Chat History

"""
            for i, msg in enumerate(self.session_manager.chat_history, 1):
                msg_type = msg["type"].title()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                context_md += f"### {i}. {msg_type}\n{content}\n\n"

            self.console.print(Panel(Markdown(context_md), title="[bold]Agent Context (Markdown)[/bold]", border_style="blue"))

    def show_settings(self):
        """Show configuration file locations and content."""
        settings_text = Text()
        settings_text.append("Configuration Files:\n\n", style="bold")

        settings_text.append(f"Settings: {self.config_manager.settings_file}\n", style="cyan")
        settings_text.append(f"Sessions: {self.config_manager.sessions_dir}\n", style="cyan")
        settings_text.append(f"History: {self.config_manager.app_dir / 'input_history.txt'}\n\n", style="cyan")

        settings_text.append("Current Settings:\n", style="bold")

        # Show non-sensitive settings
        safe_settings = {}
        for key, value in self.config_manager.settings.items():
            if key in ["api_key"]:
                safe_settings[key] = "***" if value else "Not set"
            else:
                safe_settings[key] = value

        settings_json = json.dumps(safe_settings, indent=2)
        settings_text.append(settings_json, style="dim")

        self.console.print(Panel(settings_text, title="[bold]Settings[/bold]", border_style="yellow"))

    def process_shell_command(self, command: str):
        """Process shell commands starting with !"""
        import subprocess

        try:
            # Remove the ! prefix
            shell_command = command.lstrip().lstrip("!").strip()

            # Show what we're running
            self.session_manager.add_message("system", f"Running shell command: {shell_command}", "Shell Command")

            # Execute the command
            result = subprocess.run(
                shell_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Display output
            if result.stdout:
                self.session_manager.add_message("tool", result.stdout, f"Output: {shell_command}")
            if result.stderr:
                self.session_manager.add_message("error", result.stderr, f"Error: {shell_command}")

            if result.returncode != 0:
                self.session_manager.add_message("error", f"Command failed with exit code {result.returncode}", f"Failed: {shell_command}")

        except subprocess.TimeoutExpired:
            self.session_manager.add_message("error", "Command timed out after 30 seconds", f"Timeout: {shell_command}")
        except Exception as e:
            self.session_manager.add_message("error", f"Error executing command: {str(e)}", f"Error: {shell_command}")

    def process_user_message(self, user_message: str):
        """Process a user message and get agent response."""
        if not user_message.strip():
            return

        # Handle shell commands
        if user_message.startswith("!"):
            self.process_shell_command(user_message)
            return

        # Add user message to history
        self.session_manager.add_message("user", user_message)

        # Update display
        if self.live:
            self.live.update(self.render_chat_history())

        try:
            # Show thinking indicator without interfering with Live display
            thinking_text = Text("⠋ Agent is thinking...", style="cyan")
            self.console.print(thinking_text, end="\r")

            # Get agent response
            response = self.agent_manager.run_agent(user_message)

            # Clear the thinking indicator
            self.console.print(" " * len(str(thinking_text)), end="\r")
            self.console.print()  # Add a newline for proper spacing

            # Add agent response to history
            if hasattr(response, 'answer'):
                self.session_manager.add_message("agent", response.answer)
            else:
                self.session_manager.add_message("agent", str(response))

            # Handle tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "Unknown tool")
                    tool_args = tool_call.get("arguments", {})
                    self.session_manager.add_message(
                        "tool",
                        f"Called {tool_name} with {tool_args}",
                        f"Tool: {tool_name}"
                    )

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            get_logger().error(error_msg)
            self.session_manager.add_message("error", error_msg)

        # Update display
        if self.live:
            self.live.update(self.render_chat_history())

    def run(self):
        """Main application loop."""
        self.running = True
        self.live = None  # We don't use Live display to avoid conflicts

        # Display welcome
        self.display_welcome()

        # Show initial chat history
        if self.session_manager.chat_history:
            self.console.print("\n[bold]Previous conversation:[/bold]")
            self.console.print(self.render_chat_history())

        # Simple input loop without Live display
        while True:
            try:
                # Get user input
                user_input = self.input_handler.get_input("> ")

                if not user_input.strip():
                    continue

                # Handle shell commands first (check for both ! and escaped \!)
                if user_input.startswith("!") or user_input.startswith(r"\!"):
                    # Remove escape backslash if present
                    if user_input.startswith(r"\!"):
                        user_input = "!" + user_input[2:]
                    self.process_shell_command(user_input)
                    # Show updated chat history
                    self.console.print("\n[bold]Updated Chat:[/bold]")
                    self.console.print(self.render_chat_history())
                    continue

                # Handle commands
                if self.handle_command(user_input):
                    if not self.running:  # /quit was called
                        break
                    continue

                # Process regular message
                self.process_user_message(user_input)

                # Show updated chat history
                self.console.print("\n[bold]Updated Chat:[/bold]")
                self.console.print(self.render_chat_history())

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit.[/yellow]")
            except EOFError:
                break
            except Exception as e:
                get_logger().error(f"Unexpected error: {e}")
                self.console.print(f"[red]Error: {e}[/red]")

        # Save session on exit
        self.session_manager.save_session()
        self.console.print("\n[bold green]Goodbye! 👋[/bold green]")


def main():
    """Main entry point."""
    # Set up platform-specific directories
    APP_NAME = "vibeagent"
    config_dir = Path(platformdirs.user_config_dir(APP_NAME))
    log_dir = Path(platformdirs.user_log_dir(APP_NAME))
    data_dir = Path(platformdirs.user_data_dir(APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Suppress specific warnings to keep terminal clean
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Parameter 'structured_output' was not specified.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Currently it defaults to False, but in version 1.25.*")

    # Set up logging early to capture all startup messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_dir / "vibeagent.log",
        filemode="a",
    )

    # Log session start
    logger = logging.getLogger(__name__)
    get_logger().info("--- VibeAgent session started, logging to vibeagent.log ---")

    parser = argparse.ArgumentParser(description="vibeagent - Rich Chat Interface")
    parser.add_argument("--model", help="Override model setting")
    parser.add_argument("--api-key-env-var", help="Environment variable containing API key")
    parser.add_argument("--api-base", help="Override API base URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load environment variables
    load_dotenv()

    try:
        app = RichChatApp()

        # Apply command line overrides
        if args.model:
            app.config_manager.set("model", args.model)
            app.agent_manager.initialize_agent()

        if args.api_key_env_var:
            api_key = os.getenv(args.api_key_env_var)
            if api_key:
                app.config_manager.set("api_key", api_key)
                app.agent_manager.initialize_agent()

        if args.api_base:
            app.config_manager.set("api_base", args.api_base)
            app.agent_manager.initialize_agent()

        app.run()

    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)
    except Exception as e:
        get_logger().error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
