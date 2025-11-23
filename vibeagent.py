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
import subprocess
import select
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from rich.prompt import Prompt, Confirm
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter, Completer, Completion
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

class CommandCompleter(Completer):
    """Custom completer for commands and their arguments."""

    def __init__(self, base_commands, model_provider=None, session_provider=None):
        self.base_commands = base_commands
        self.model_provider = model_provider
        self.session_provider = session_provider

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        # If empty or just starting, complete base commands
        if " " not in text:
            for cmd in self.base_commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        # Check for command arguments
        parts = text.split(" ", 1)
        command = parts[0].lower()
        arg_prefix = parts[1] if len(parts) > 1 else ""

        if command == "/model" and self.model_provider:
            models = self.model_provider()
            for model in models:
                if model.lower().startswith(arg_prefix.lower()):
                    yield Completion(model, start_position=-len(arg_prefix))
        
        elif command in ["/save", "/load", "/delete"] and self.session_provider:
            sessions = self.session_provider()
            for session in sessions:
                if session.lower().startswith(arg_prefix.lower()):
                    yield Completion(session, start_position=-len(arg_prefix))
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

        # Initialize with empty providers, will be set later
        self.completer = CommandCompleter(self.base_commands)

        self.session = PromptSession(
            history=FileHistory(str(self.history_file)),
            completer=self.completer,
            key_bindings=self.key_bindings,
            multiline=False,
            wrap_lines=True
        )

    def set_providers(self, model_provider=None, session_provider=None):
        """Set providers for dynamic completion."""
        self.completer.model_provider = model_provider
        self.completer.session_provider = session_provider

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
                # Suggest available models
                models = self.config_manager.get("available_models", []) or []
                # Also try to get from agent manager if possible (circular dependency issue otherwise)
                # For now, we'll rely on what's in settings or hardcoded common ones if empty

                return [f"/model {model}" for model in models]
            elif command == "/compress":
                strategies = ["drop_oldest", "middle_out", "summarize"]
                return [f"/compress {strategy}" for strategy in strategies]
            elif command == "/dump-context":
                formats = ["markdown", "json", "text"]
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



class ShellSession:
    """Manages a persistent shell session."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.process = None
        self.delimiter = f"VIBEAGENT_END_{uuid.uuid4().hex[:8]}"
        self._initialize_session()

    def get_cwd(self) -> str:
        """Get current working directory of the shell session."""
        output, exit_code = self.run_command("pwd")
        if exit_code == 0:
            return output.strip()
        return os.getcwd()

    def _initialize_session(self):
        """Start the persistent bash process."""
        command = ["/bin/bash"]
        
        if self.config_manager:
            containers_config = self.config_manager.get("containers", {})
            if containers_config.get("enabled", False) and containers_config.get("sandboxBangShellCommands", False):
                engine = containers_config.get("engine", "apptainer")
                image = containers_config.get("image")
                home_mount = containers_config.get("home_mount_point", "/home/agent")
                allowed_paths = self.config_manager.get("allowedPaths", [os.getcwd()])
                cwd = os.getcwd()

                if engine == "apptainer" and image:
                    # Auto-prefix docker:// if likely a registry image
                    if not any(image.startswith(p) for p in ["docker://", "library://", "shub://", "oras://", "/", "./"]):
                        image = f"docker://{image}"
                        
                    command = ["apptainer", "--silent", "exec", "--cleanenv", "--no-home", "--no-mount", "hostfs"]
                    for path in allowed_paths:
                        command.extend(["--bind", path])
                    command.extend(["--cwd", cwd, image, "/bin/bash"])
                    get_logger().info(f"Using Apptainer shell: {' '.join(command)}")
                    
                elif engine == "docker" and image:
                    uid = os.getuid()
                    gid = os.getgid()
                    
                    # Mount allowed paths and CWD
                    mounts = []
                    paths_to_mount = set(allowed_paths)
                    paths_to_mount.add(cwd)
                    
                    for path in paths_to_mount:
                        mounts.extend(["-v", f"{path}:{path}"])

                    command = [
                        "docker", "run", "-i", "--rm",
                        "-u", f"{uid}:{gid}",
                        "-w", cwd,
                    ]
                    command.extend(mounts)
                    command.extend([image, "/bin/bash"])
                    
                    get_logger().info(f"Using Docker shell: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                preexec_fn=os.setsid  # New session group
            )
            os.set_blocking(self.process.stdout.fileno(), False)
            os.set_blocking(self.process.stderr.fileno(), False)
            get_logger().info("Started persistent shell session")
        except Exception as e:
            get_logger().error(f"Failed to start shell session: {e}")

    def run_command(self, command: str, timeout: int = 30) -> tuple[str, int]:
        """Run a command in the persistent shell and return output and exit code."""
        if not self.process or self.process.poll() is not None:
            self._initialize_session()

        if not self.process:
            return "Error: Shell session not active", 1

        try:
            # Prepare command with delimiter to detect end
            full_command = f"{command}; echo; echo '{self.delimiter}':$?\n"
            
            # Write to stdin
            self.process.stdin.write(full_command)
            self.process.stdin.flush()

            output_lines = []
            stdout_buffer = ""
            stderr_buffer = ""
            exit_code = 0
            start_time = time.time()
            
            while True:
                if time.time() - start_time > timeout:
                    return "".join(output_lines) + f"\nError: Command timed out after {timeout} seconds", 124

                reads = [self.process.stdout.fileno(), self.process.stderr.fileno()]
                ret = select.select(reads, [], [], 0.1)

                if not ret[0]:
                    if self.process.poll() is not None:
                        break
                    continue

                for fd in ret[0]:
                    if fd == self.process.stdout.fileno():
                        try:
                            chunk = os.read(fd, 4096).decode('utf-8', errors='replace')
                        except OSError:
                            chunk = ""
                            
                        if not chunk:
                            break
                        
                        stdout_buffer += chunk
                        while '\n' in stdout_buffer:
                            line, stdout_buffer = stdout_buffer.split('\n', 1)
                            if self.delimiter in line:
                                try:
                                    _, code = line.strip().split(":")
                                    exit_code = int(code)
                                except ValueError:
                                    pass
                                return "".join(output_lines), exit_code
                            output_lines.append(line + '\n')
                    
                    elif fd == self.process.stderr.fileno():
                        try:
                            chunk = os.read(fd, 4096).decode('utf-8', errors='replace')
                        except OSError:
                            chunk = ""
                            
                        if chunk:
                            stderr_buffer += chunk
                            while '\n' in stderr_buffer:
                                line, stderr_buffer = stderr_buffer.split('\n', 1)
                                output_lines.append(line + '\n')

            return "".join(output_lines), exit_code

        except Exception as e:
            get_logger().error(f"Error running shell command: {e}")
            return f"Error: {str(e)}", 1

    def close(self):
        """Close the shell session."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                self.process.kill()
            self.process = None


class AgentManager:
    """Manages agent initialization and communication."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agent = None
        self.shell_session: 'ShellSession | None' = None
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

    def fetch_models(self):
        """Fetch available models from configured endpoints."""
        from openai import OpenAI
        
        endpoints = self.config_manager.settings.get("endpoints", {})
        if not endpoints:
            return

        for provider_name, config in endpoints.items():
            if not config.get("enabled", True):
                get_logger().info(f"Skipping disabled endpoint: {provider_name}")
                continue
                
            try:
                api_key = self._substitute_env_vars(config.get("api_key", ""))
                api_base = config.get("api_base")
                
                client = OpenAI(api_key=api_key, base_url=api_base)
                models = client.models.list()
                
                for model in models.data:
                    model_id = f"{provider_name}:{model.id}"
                    self.model_details[model_id] = {
                        "provider": provider_name,
                        "original_id": model.id,
                        "api_key": api_key,
                        "api_base": api_base
                    }
                    if model_id not in self.available_models:
                        self.available_models.append(model_id)
                        
                get_logger().info(f"Fetched {len(models.data)} models from {provider_name}")
                
            except Exception as e:
                get_logger().warning(f"Failed to fetch models from {provider_name}: {e}")

    def initialize_agent(self):
        """Initialize the smolagents agent with proper configuration."""
        try:
            get_logger().info("Initializing agent with configuration...")

            # Fetch available models first
            self.fetch_models()

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

        @tool
        def run_shell_command(command: str) -> str:
            """
            Run a shell command and return the output.
            Args:
                command: The shell command to run.
            """
            if not hasattr(self, 'shell_session') or not self.shell_session:
                return "Error: Shell session not available."
            
            output, exit_code = self.shell_session.run_command(command)
            return f"Exit Code: {exit_code}\nOutput:\n{output}"

        return [aider_edit_file, run_shell_command]

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
        self.shell_session = ShellSession(self.config_manager)
        
        # Link shell session to agent manager
        self.agent_manager.shell_session = self.shell_session

        # Initialize Telemetry
        if TELEMETRY_AVAILABLE:
            try:
                # Configure Phoenix OTEL
                register(
                    project_name="vibeagent",
                    endpoint="http://localhost:6006/v1/traces"
                )
                
                # Initialize instrumentor
                SmolagentsInstrumentor().instrument()
                
                # Configure tracer provider
                tracer_provider = TracerProvider()
                tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                trace.set_tracer_provider(tracer_provider)
                
                get_logger().info("Telemetry initialized successfully")
            except Exception as e:
                get_logger().warning(f"Failed to initialize telemetry: {e}")

        # Set providers for autocomplete
        self.input_handler.set_providers(
            model_provider=lambda: list(self.agent_manager.model_details.keys()),
            session_provider=self._get_session_names
        )

        self.running = False
        self.live = None

        # Load or create session
        session_id = self.config_manager.get("session_id")
        if session_id and self.session_manager.load_session(session_id):
            pass  # Session loaded successfully
        else:
            self.session_manager.new_session()

    def _get_session_names(self):
        """Get list of available session names for autocomplete."""
        sessions = self.session_manager.list_sessions()
        # Return both IDs and custom names if available
        names = []
        custom_sessions_dir = self.config_manager.app_dir / "custom_sessions"
        if custom_sessions_dir.exists():
            for f in custom_sessions_dir.glob("*.json"):
                names.append(f.stem)
        return names

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

    def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        if not command.startswith('/'):
            return False

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
            if arg:
                self.handle_save_command(arg)
            else:
                self.console.print("[red]Usage: /save <name>[/red]")
            return True
        elif cmd == '/load':
            if arg:
                self.handle_load_command(arg)
            else:
                self.console.print("[red]Usage: /load <name>[/red]")
            return True
        elif cmd == '/delete':
            if arg:
                self.handle_delete_command(arg)
            else:
                self.console.print("[red]Usage: /delete <name>[/red]")
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
            mcp_table.add_column("Server", style="green", min_width=15)
            mcp_table.add_column("Status", style="white", width=12)
            mcp_table.add_column("Command", style="dim", min_width=30, overflow="fold")

            # Sort servers: enabled first, then disabled
            enabled_servers = []
            disabled_servers = []

            for server_name, server_config in mcp_servers.items():
                enabled = server_config.get("enabled", False)
                if enabled:
                    enabled_servers.append((server_name, server_config))
                else:
                    disabled_servers.append((server_name, server_config))

            # Process enabled servers first, then disabled
            sorted_servers = enabled_servers + disabled_servers

            for server_name, server_config in sorted_servers:
                enabled = server_config.get("enabled", False)
                status = "✅ Enabled" if enabled else "❌ Disabled"
                command = server_config.get("command", "N/A")
                args = server_config.get("args", [])

                # Combine command and args for full command display
                if command != "N/A" and args:
                    full_command = f"{command} {' '.join(args)}"
                elif command != "N/A":
                    full_command = command
                else:
                    full_command = "N/A"

                mcp_table.add_row(server_name, status, full_command)

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

            # Show current model
            self.console.print(f"\n[cyan]Current model: {current_model}[/cyan]")

    def refresh_models(self):
        """Refresh model list from API (placeholder for now)."""
        self.console.print("[yellow]Model refresh not yet implemented in Rich version[/yellow]")
        self.console.print(f"[cyan]Current model: {self.config_manager.get('model', 'Unknown')}[/cyan]")

    def _count_tokens_with_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken with cl100k_base encoding (GPT-4/3.5 compatible)."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to rough approximation if tiktoken fails
            return len(text) // 4

    def compress_context(self, strategy: str = None):
        """Compress conversation context."""
        if not strategy:
            strategy = "summarize"  # Default strategy

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
            if not self.agent_manager.agent:
                 self.console.print("[red]Agent is not initialized. Cannot summarize.[/red]")
                 return

            progress_console = Console(file=sys.__stdout__)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Summarizing and compressing context...[/bold green]"),
                TimeElapsedColumn(),
                console=progress_console,
                transient=True,
            ) as progress:
                progress.add_task("", total=None)
                try:
                    messages = self.agent_manager.agent.write_memory_to_messages()
                    history_text = "\n".join(
                        [
                            f"{m.role.value}: {''.join(c['text'] if isinstance(c, dict) and 'text' in c else str(c) for c in (m.content if isinstance(m.content, list) else [m.content]))}"
                            for m in messages
                        ]
                    )
                    summary_prompt = f"""
                    Summarize the following conversation history to retain the full context of the session.
                    Structure your summary to cover:
                    1. The START: The user's initial intent and original questions.
                    2. The MIDDLE: Key steps taken, tools used, and intermediate results or discoveries.
                    3. The END: The final answer, solution, or current state.

                    CRITICAL REQUIREMENTS:
                    - DON'T summarize the tools available, but DO summarize which tools were used and why.
                    - ALWAYS include the actual content of the final_answer.
                    - If search results were returned, keep the actual results details.
                    - If the user has been generating code or files, include the final version of the code or final file path(s).
                    - Preserve any specific constraints or preferences expressed by the user throughout the conversation.

                    Conversation History:
                    {history_text}
                    """.strip()

                    # Get current model details to create a summarizer
                    current_model_id = self.config_manager.get("defaultModel")
                    if not current_model_id or current_model_id not in self.agent_manager.model_details:
                        # Fallback to first available if default not found (shouldn't happen if agent is init)
                        if self.agent_manager.available_models:
                            current_model_id = self.agent_manager.available_models[0]
                        else:
                            raise Exception("No model details available for summarization")
                    
                    model_details = self.agent_manager.model_details[current_model_id]
                
                    summarizer_model = OpenAIServerModel(
                        model_id=model_details["original_id"],
                        api_key=model_details["api_key"],
                        api_base=model_details["api_base"],
                    )

                    # Use direct model call
                    messages = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=[
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant that summarizes conversations.",
                                }
                            ],
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": summary_prompt}],
                        ),
                    ]

                    response = summarizer_model.generate(messages)
                    summary = response.content
                    if isinstance(summary, list):
                        summary = "".join(
                            c.get("text", "")
                            for c in summary
                            if isinstance(c, dict) and "text" in c
                        )
                    elif isinstance(summary, str):
                        summary = summary
                    else:
                        summary = str(summary)

                    if not summary or summary.strip() == "":
                        self.console.print("[red]Warning: Summarization returned empty result. Using fallback summary.[/red]")
                        summary = f"Conversation summary: Previous conversation has been compressed. Key information has been preserved."

                    # Calculate token count
                    summary_tokens = self._count_tokens_with_tiktoken(summary) if summary else 0

                    # Create a simple summary step
                    timing = Timing(start_time=time.time())
                    timing.end_time = time.time()

                    token_usage = TokenUsage(
                        input_tokens=summary_tokens,
                        output_tokens=summary_tokens,
                    )

                    summary_step = ActionStep(
                        step_number=1,
                        timing=timing,
                        model_output=summary,
                        observations=None,
                        token_usage=token_usage,
                    )

                    # Replace agent memory
                    self.agent_manager.agent.memory.steps = [summary_step]
                    self.agent_manager.agent.step_number = 2
                    
                    self.console.print(f"[green]Context summarized successfully. New summary token count: {summary_tokens}[/green]")
                    
                    # Also clear the chat history in the UI/session to reflect the compression
                    # We'll keep the last message or just a system message saying it was compressed
                    self.session_manager.chat_history = [
                        {
                            "type": "system",
                            "content": f"Conversation history compressed. Summary:\n{summary}",
                            "timestamp": datetime.now().isoformat()
                        }
                    ]

                except Exception as e:
                    self.console.print(f"[red]Error during summarization: {e}[/red]")
        else:
            self.console.print("[yellow]Not enough messages to compress[/yellow]")

    def dump_context(self, format_type: str = "markdown"):
        """Display agent's current memory/context."""
        if format_type not in ["markdown", "json", "text"]:
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
        elif format_type == "text":
            # Raw text format
            context_text = f"Agent Context\n\n"
            context_text += f"Session ID: {self.session_manager.session_id}\n"
            context_text += f"Model: {self.config_manager.get('model')}\n"
            context_text += f"Messages: {len(self.session_manager.chat_history)}\n\n"
            context_text += "Chat History:\n\n"
            
            for i, msg in enumerate(self.session_manager.chat_history, 1):
                msg_type = msg["type"].title()
                content = msg["content"]
                context_text += f"[{i}. {msg_type}]\n{content}\n\n"
            
            self.console.print(Panel(Text(context_text), title="[bold]Agent Context (Text)[/bold]", border_style="blue"))
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
                content = msg["content"]
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

    def _handle_shell_command(self, command: str):
        """Process shell commands starting with !"""
        try:
            # Remove the ! prefix
            shell_command = command.lstrip().lstrip("!").strip()

            # Show what we're running
            self.session_manager.add_message("user", f">!{shell_command}")
            self.console.print(f"[bold yellow]> !{shell_command}[/bold yellow]")

            # Execute the command via persistent session
            output, exit_code = self.shell_session.run_command(shell_command)
            
            # Store result for redirection
            self.last_shell_result = {
                'command': shell_command,
                'stdout': output, # ShellSession combines stdout/stderr currently, but we can split if needed later or just use output
                'stderr': "", # ShellSession combines them
                'exit_code': exit_code
            }

            # Display output
            if output:
                self.session_manager.add_message("tool", output, f"Output: {shell_command}")
                self.console.print(Panel(output.strip(), title="Shell Output", border_style="yellow"))
            
            if exit_code != 0:
                error_msg = f"Command exited with code {exit_code}"
                self.session_manager.add_message("error", error_msg, f"Failed: {shell_command}")
                self.console.print(f"[red]{error_msg}[/red]")

        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            self.session_manager.add_message("error", error_msg, f"Error: {shell_command}")
            self.console.print(f"[red]{error_msg}[/red]")

    def run_agent(self, user_message: str):
        """Run the agent with a loading spinner."""
        # Use a separate console writing to sys.__stdout__ to bypass redirection
        progress_console = Console(file=sys.__stdout__)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}[/bold green]"),
            TimeElapsedColumn(),
            console=progress_console,
            transient=True
        ) as progress:
            task = progress.add_task("Agent is thinking...", total=None)
            
            # Get CWD inside the spinner
            progress.update(task, description="Checking environment...")
            try:
                cwd = self.shell_session.get_cwd()
                system_note = f"[System Note: Current Working Directory is {cwd}]"
                full_message = f"{system_note}\n\n{user_message}"
            except Exception as e:
                get_logger().warning(f"Failed to get CWD: {e}")
                full_message = user_message

            progress.update(task, description="Agent is thinking...")
            return self.agent_manager.run_agent(full_message)

    def _handle_redirect_command(self, user_message: str) -> bool:
        """Handle redirect commands (!>, !1>, !2>). Returns True if handled."""
        import re
        redirect_match = re.match(r"^!\s*([12]?)\s*>", user_message)
        
        # Fallback check for simple cases
        simple_redirect = False
        if not redirect_match:
            clean_msg = user_message.replace(" ", "")
            if clean_msg.startswith(("!>", "!1>", "!2>")):
                simple_redirect = True

        if not (redirect_match or simple_redirect):
            return False

        if not hasattr(self, 'last_shell_result') or not self.last_shell_result:
            self.console.print("[red]No previous shell command to redirect output from.[/red]")
            return True

        if simple_redirect:
            # Manual parsing fallback
            if user_message.lstrip().startswith("!1>"):
                redirect_type_num = "1"
                user_comment = user_message.lstrip()[3:].strip()
            elif user_message.lstrip().startswith("!2>"):
                redirect_type_num = "2"
                user_comment = user_message.lstrip()[3:].strip()
            else: # !>
                redirect_type_num = ""
                user_comment = user_message.lstrip()[2:].strip()
        else:
            assert redirect_match is not None
            redirect_type_num = redirect_match.group(1) # "" or "1" or "2"
            user_comment = user_message[redirect_match.end():].strip()
        
        # Construct prompt
        prompt = f"{user_comment}\n\n" if user_comment else ""
        prompt += f"Command: {self.last_shell_result['command']}\n"
        prompt += f"Exit Code: {self.last_shell_result['exit_code']}\n"
        
        output = self.last_shell_result['stdout']
        
        if redirect_type_num == "" or redirect_type_num == "1":
                prompt += f"\nOutput:\n{output}\n"
        
        # Add to history as a user message with the context
        self.session_manager.add_message("user", f"{user_message}\n\n[Redirected Output]\n{output}")
        
        # Run agent
        response = self.run_agent(prompt)
        self._handle_agent_response(response)
        return True

    def _process_chat_message(self, user_message: str):
        """Process a regular chat message."""
        user_message = user_message.strip()
        if not user_message:
            return

        # Add user message to history
        self.session_manager.add_message("user", user_message)

        # Update display
        if self.live:
            self.live.update(self.render_chat_history())

        try:
            # Get agent response
            response = self.run_agent(user_message)
            self._handle_agent_response(response)
            return response
            
        except Exception as e:
            error_msg = f"Error running agent: {str(e)}"
            get_logger().error(error_msg)
            self.session_manager.add_message("error", error_msg)
            self.console.print(self.message_renderer.render_message("error", error_msg))
            return None

    def dispatch_message(self, user_input: str) -> bool:
        """
        Central message dispatcher.
        Returns True if the application should continue running, False otherwise.
        """
        if not user_input.strip():
            return True

        # 1. Slash Commands
        if user_input.startswith("/"):
            if self._handle_slash_command(user_input):
                return self.running # _handle_slash_command sets self.running to False on /quit

        # 2. Redirect Commands
        if user_input.startswith("!"):
            if self._handle_redirect_command(user_input):
                return True
            
            # 3. Shell Commands (if not a redirect)
            self._handle_shell_command(user_input)
            return True

        # 4. Chat Messages
        self._process_chat_message(user_input)
        return True

    def _handle_agent_response(self, response):
        """Handle agent response including tool calls and content display."""
        if not response:
            return

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
        
        # Assuming 'response' object has a 'content' attribute for the agent's main message
        content = getattr(response, 'content', str(response))
        self.session_manager.add_message("agent", content)
        self.console.print(self.message_renderer.render_message("agent", content))

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

                # Dispatch message
                if not self.dispatch_message(user_input):
                    # dispatch_message returns False if we should exit (e.g. /quit)
                    self.cleanup()
                    break
                continue

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit.[/yellow]")
            except EOFError:
                # Clean up on Ctrl+D
                self.console.print("\n[yellow]Received EOF. Exiting...[/yellow]")
                self.cleanup()
                break
            except Exception as e:
                get_logger().error(f"Unexpected error: {e}")
                self.console.print(f"[red]Error: {e}[/red]")
                # Clean up even on error
                self.cleanup()
                break

        # Save session on exit
        self.session_manager.save_session()
        self.console.print("\n[bold green]Goodbye! 👋[/bold green]")

        # Clean up terminal state
        self.cleanup()

    def cleanup(self):
        """Clean up terminal state and resources."""
        try:
            # Stop Live display if running
            if self.live:
                self.live.stop()
                self.live = None

            # Clean up prompt-toolkit session
            if hasattr(self.input_handler, 'session') and self.input_handler.session:
                try:
                    self.input_handler.session.app.exit()
                except:
                    pass  # Ignore cleanup errors

            # Reset Rich console
            self.console.clear()
            self.console.show_cursor(True)

            # Reset terminal attributes (without clearing screen)
            sys.stdout.write('\033[0m')  # Reset all attributes
            sys.stdout.write('\033[?25h')  # Ensure cursor is visible
            sys.stdout.write('\n')  # Move to new line
            sys.stdout.flush()

        except Exception as e:
            # If cleanup fails, just try a basic reset
            try:
                sys.stdout.write('\033[0m')  # Reset all attributes
                sys.stdout.write('\033[?25h')  # Ensure cursor is visible
                sys.stdout.flush()
            except:
                pass


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

    app = None
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
        if app:
            app.cleanup()
        print("\nGoodbye! 👋")
        sys.exit(0)
    except Exception as e:
        if app:
            app.cleanup()
        get_logger().error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
