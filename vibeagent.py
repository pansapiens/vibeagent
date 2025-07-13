# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "smolagents[litellm,mcp,telemetry,toolkit]",
#   "textual",
#   "python-dotenv",
#   "aider-chat",
#   "openai",
#   "textual-autocomplete",
#   "platformdirs",
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
from functools import partial
from pathlib import Path
from dotenv import load_dotenv
from contextlib import redirect_stdout
import platformdirs
from smolagents import ToolCallingAgent, tool, MCPClient
from smolagents.models import OpenAIServerModel, MessageRole, ChatMessage, TokenUsage
from smolagents.monitoring import Timing
from mcp import StdioServerParameters
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import (
    Header,
    Footer,
    Input,
    Static,
    Markdown,
    OptionList,
    LoadingIndicator,
)
from textual.widgets.option_list import Option
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.command import Provider, Hit, Hits
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from textual_autocomplete import AutoComplete, DropdownItem
from textual_autocomplete._autocomplete import TargetState
from smolagents.memory import ActionStep, MemoryStep


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


cyberpunk_theme = Theme(
    name="cyberpunk",
    primary="#00FFFF",  # Electric Blue
    secondary="#FF00FF",  # Vivid Pink
    accent="#CC00CC",  # Deep Purple
    warning="#FFA500",  # Orange
    error="#FF4500",  # OrangeRed
    success="#00FF00",  # Luminescent Green
    foreground="#E0E0E0",
    background="#101010",
    surface="#1A1A1A",
    panel="#252526",
    dark=True,
    variables={
        "border": "#00FFFF80",
        "border-blurred": "#00FFFF4D",
        "block-cursor-foreground": "#101010",
        "block-cursor-background": "#00FFFF",
        "input-cursor-background": "#00FFFF",
        "input-cursor-foreground": "#101010",
        "input-selection-background": "#FF00FF40",
        "scrollbar-color": "#252526",
        "scrollbar-color-hover": "#FF00FF",
        "scrollbar-color-active": "#CC00CC",
        "footer-key-background": "#00FFFF",
        "footer-key-foreground": "#101010",
        "link-color": "#00FFFF",
        "link-style": "underline",
        "link-background-hover": "#00FFFF",
        "link-color-hover": "#101010",
        "link-style-hover": "bold not underline",
    },
)


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
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# Load environment variables from .env file
load_dotenv()


class ModelSelectScreen(ModalScreen[str]):
    """Screen for selecting a model."""

    def __init__(
        self,
        models: list[str],
        favorites: list[str],
        current_model: str,
        model_details: dict,
    ) -> None:
        super().__init__()
        self.all_models = models
        self.favorites = favorites
        self.current_model = current_model
        self.model_details = model_details

    def compose(self) -> ComposeResult:
        # separate favorite models and other models
        fav_options = []
        other_options = []

        # Create options, favorites first
        for model_id in self.all_models:
            # Use display_id for the list, but check against original_id for favorites.
            details = self.model_details.get(model_id)
            if not details:
                # Should not happen if fetch_models is working correctly, but good to be safe
                other_options.append(Option(model_id, id=model_id))
                continue

            original_id = details.get("original_id", model_id)
            if original_id in self.favorites:
                fav_options.append(Option(f"{model_id} (favorite)", id=model_id))
            else:
                other_options.append(Option(model_id, id=model_id))

        # Sort favorites alphabetically and others alphabetically
        fav_options.sort(key=lambda o: o.prompt)
        other_options.sort(key=lambda o: o.prompt)

        with Vertical(id="model-select-dialog"):
            yield Static("Select a Model", classes="dialog-title")
            yield OptionList(*fav_options, *other_options, id="model-select")

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        # Try to highlight the current model.
        try:
            options = option_list.query(Option)
            for i, option in enumerate(options):
                if option.id == self.current_model:
                    option_list.highlighted = i
                    break
        except Exception:
            pass  # just don't highlight
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Called when an option is selected."""
        self.dismiss(str(event.option.id))


class ModelSelectProvider(Provider):
    """A command provider for selecting the model."""

    async def search(self, query: str) -> Hits:
        """Search for the select model command."""
        matcher = self.matcher(query)
        command = "Select model"
        score = matcher.match(command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(command),
                self.app.action_select_model,
                help="Choose a different LLM to chat with.",
            )


class ChatApp(App):
    """A textual-based chat interface for a smolagents agent."""

    CSS = """
    Screen {
        background: $background;
        color: $foreground;
    }

    #chat-history {
        padding: 1;
        background: $surface;
    }

    .user-message {
        background: $panel;
        color: $foreground;
        padding: 1;
        margin-bottom: 1;
        border-left: thick $primary;
    }

    .agent-message {
        background: $panel;
        color: $foreground;
        padding: 1;
        margin-bottom: 1;
        border-left: thick $secondary;
        text-wrap: wrap;
        text-overflow: fold;
    }

    .agent-thinking {
        background: transparent;
        color: $primary;
        height: 1;
        padding: 0;
        margin-top: 1;
        margin-bottom: 1;
    }

    .tool-call-message {
        color: $success;
        padding-left: 1;
        margin-bottom: 1;
    }

    .info-message {
        background: $panel;
        color: $primary-muted;
        border-left: thick $primary-muted;
        padding: 1;
        margin-bottom: 1;
        text-wrap: wrap;
        text-overflow: fold;
    }

    .error-message {
        background: $panel;
        color: $error;
        padding: 1;
        margin-bottom: 1;
        border-left: thick $error;
        text-wrap: wrap;
        text-overflow: fold;
    }

    Input {
        background: $panel;
        color: $foreground;
        border: tall $primary 30%;
    }

    Input:focus {
        border: tall $primary;
    }

    #model-select-dialog {
        background: $surface;
        padding: 1 2;
        border: thick $accent;
        width: 80;
        height: auto;
        align: center middle;
    }

    .dialog-title {
        align: center top;
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
        color: $primary;
    }

    #model-select OptionList {
        background: $background;
    }

    #model-select Option:hover {
        background: $primary 20%;
    }

    #model-select Option.--highlight {
        background: $primary;
        color: $background;
    }

    Footer {
        background: $panel;
    }

    Footer > .footer--key {
        background: $footer-key-background;
        color: $footer-key-foreground;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("up", "history_prev", "Previous command"),
        ("down", "history_next", "Next command"),
        ("pageup", "scroll_up", "Scroll History Up"),
        ("pagedown", "scroll_down", "Scroll History Down"),
        ("escape", "cancel_request", "Cancel Request"),
    ]
    COMMANDS = App.COMMANDS | {ModelSelectProvider}

    # Characters from Unicode's "Combining Diacritical Marks" block
    # as mentioned in https://en.wikipedia.org/wiki/Zalgo_text
    ZALGO_UP = [
        "\u030d",
        "\u030e",
        "\u0304",
        "\u0305",
        "\u033f",
        "\u0311",
        "\u0306",
        "\u0310",
        "\u0352",
        "\u0357",
        "\u0351",
        "\u0307",
        "\u0308",
        "\u030a",
        "\u0342",
        "\u0343",
        "\u0344",
        "\u034a",
        "\u034b",
        "\u034c",
        "\u0303",
        "\u0302",
        "\u030c",
        "\u0350",
        "\u0300",
        "\u0301",
        "\u030b",
        "\u030f",
        "\u0312",
        "\u0313",
        "\u0314",
        "\u033d",
        "\u0309",
        "\u0363",
        "\u0364",
        "\u0365",
        "\u0366",
        "\u0367",
        "\u0368",
        "\u0369",
        "\u036a",
        "\u036b",
        "\u036c",
        "\u036d",
        "\u036e",
        "\u036f",
        "\u033e",
        "\u035b",
        "\u0346",
        "\u031a",
    ]
    ZALGO_DOWN = [
        "\u0316",
        "\u0317",
        "\u0318",
        "\u0319",
        "\u031c",
        "\u031d",
        "\u031e",
        "\u031f",
        "\u0320",
        "\u0324",
        "\u0325",
        "\u0326",
        "\u0329",
        "\u032a",
        "\u032b",
        "\u032c",
        "\u032d",
        "\u032e",
        "\u032f",
        "\u0330",
        "\u0331",
        "\u0332",
        "\u0333",
        "\u0339",
        "\u033a",
        "\u033b",
        "\u033c",
        "\u0345",
        "\u0347",
        "\u0348",
        "\u0349",
        "\u034d",
        "\u034e",
        "\u0353",
        "\u0354",
        "\u0355",
        "\u0356",
        "\u0359",
        "\u035a",
        "\u0323",
    ]
    ZALGO_MID = [
        "\u0315",
        "\u031b",
        "\u0340",
        "\u0341",
        "\u0358",
        "\u0321",
        "\u0322",
        "\u0327",
        "\u0328",
        "\u0334",
        "\u0335",
        "\u0336",
        "\u034f",
        "\u035c",
        "\u035d",
        "\u035e",
        "\u035f",
        "\u0360",
        "\u0362",
        "\u0338",
        "\u0337",
        "\u0361",
        "\u0489",
    ]

    LEETSPEAK_MAP = {
        "a": "4",
        "A": "4",
        "b": "8",
        "B": "8",
        "e": "3",
        "E": "3",
        "g": "6",
        "G": "6",
        "i": "1",
        "I": "1",
        "o": "0",
        "O": "0",
        "s": "5",
        "S": "5",
        "t": "7",
        "T": "7",
        "z": "2",
        "Z": "2",
        "l": "1",
        "L": "1",
    }

    # Printable characters from various "safe" Unicode blocks above ASCII
    HIGH_ASCII_CHARS = (
        [chr(i) for i in range(0x00A1, 0x0100)]  # Latin-1 Supplement
        + [chr(i) for i in range(0x2500, 0x2580)]  # Box Drawing
        + [chr(i) for i in range(0x2588, 0x25A0)]  # Block Elements (skip some)
        + [chr(i) for i in range(0x25A0, 0x2600)]  # Geometric Shapes
    )

    class AgentResponse(Message):
        """A message containing the agent's response."""

        def __init__(self, response: str) -> None:
            super().__init__()
            self.response = response

    class ToolCall(Message):
        """A message containing the tool call name."""

        def __init__(self, tool_name: str) -> None:
            super().__init__()
            self.tool_name = tool_name

    def __init__(
        self,
        model_config: dict,
        initial_model_id: str,
        config_dir: Path,
        log_dir: Path,
    ):
        super().__init__()
        self.title = "Vibe Agent Chat"
        self.agent = None
        self.mcp_clients = []
        self.command_history = []
        self.history_index = -1
        self.settings = {}
        self.mcp_log_files = {}  # Store MCP server log files
        self.model_config = model_config
        self.model_id = initial_model_id
        self.available_models = []
        self.favorite_models = []
        self.all_tools = []
        self.tools_by_source = {}  # New
        self.instrumentor = SmolagentsInstrumentor() if TELEMETRY_AVAILABLE else None
        self.telemetry_is_active = False
        self.default_strategy = self.settings.get(
            "contextManagementStrategy", "drop_oldest"
        )
        self.global_context_length = self.settings.get("contextLength", 8192)
        self.model_context_lengths = {}
        self.model_details = {}  # Stores provider, api_key, etc. for each model
        self.is_loading = False
        self.agent_worker = None
        self.glitch_mode = False
        self.glitch_strength = 0.10
        self.config_dir = config_dir
        self.log_dir = log_dir

    def _is_phoenix_running(self) -> bool:
        import socket
        from urllib.parse import urlparse

        # Get telemetry endpoint from environment
        telemetry_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

        try:
            parsed_url = urlparse(telemetry_endpoint)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 4317

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1-second timeout
            try:
                result = sock.connect_ex((host, port))
                return result == 0
            finally:
                sock.close()
        except Exception:
            return False

    def _update_telemetry_status(self):
        if (
            not self.instrumentor
            or os.getenv("DISABLE_TELEMETRY", "false").lower() == "true"
        ):
            return

        is_phoenix_running = self._is_phoenix_running()

        if is_phoenix_running and not self.telemetry_is_active:
            try:
                logging.info("Phoenix detected. Enabling telemetry...")
                trace_provider = TracerProvider()

                # Get telemetry endpoint from environment
                telemetry_endpoint = os.getenv(
                    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
                )
                exporter = OTLPSpanExporter(endpoint=telemetry_endpoint)
                processor = BatchSpanProcessor(
                    exporter, max_queue_size=2048, schedule_delay_millis=5000
                )
                trace_provider.add_span_processor(processor)
                trace.set_tracer_provider(trace_provider)
                register()

                self.instrumentor.instrument(tracer_provider=trace_provider)
                self.telemetry_is_active = True
                logging.info(f"Telemetry enabled. Endpoint: {telemetry_endpoint}")
            except Exception as e:
                logging.error(f"Failed to enable telemetry: {e}")

        elif not is_phoenix_running and self.telemetry_is_active:
            try:
                logging.info("Phoenix not detected. Disabling telemetry...")
                self.instrumentor.uninstrument()
                self.telemetry_is_active = False
                # Also reset the global tracer provider
                trace.set_tracer_provider(trace.NoOpTracerProvider())
                logging.info("Telemetry disabled.")
            except Exception as e:
                logging.error(f"Failed to disable telemetry: {e}")

    async def fetch_models(self) -> None:
        """Fetches available models from all enabled OpenAI-compatible endpoints."""
        from openai import AsyncOpenAI

        endpoints = self.model_config.get("endpoints", {})
        if not endpoints:
            self.call_from_thread(
                self.query_one("#chat-history").mount,
                Static(
                    "No model endpoints configured in settings.",
                    classes="error-message",
                ),
            )
            return

        all_models_by_provider = {}
        model_id_counts = {}

        for provider_name, config in endpoints.items():
            if not config.get("enabled", False):
                continue
            try:
                api_key = config.get("api_key")
                api_base = config.get("api_base")
                if not api_key or not api_base:
                    logging.warning(
                        f"Skipping provider {provider_name}: missing api_key or api_base."
                    )
                    continue

                async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
                    models_response = await client.models.list()
                    provider_models = models_response.data
                    all_models_by_provider[provider_name] = provider_models
                    for model in provider_models:
                        model_id_counts[model.id] = model_id_counts.get(model.id, 0) + 1
            except Exception as e:
                self.call_from_thread(
                    self.query_one("#chat-history").mount,
                    Static(
                        f"Error fetching models from {provider_name}: {e}",
                        classes="error-message",
                    ),
                )
                logging.error(f"Error fetching models from {provider_name}: {e}")

        new_available_models = []
        new_model_context_lengths = {}
        new_model_details = {}

        for provider_name, models in all_models_by_provider.items():
            provider_config = endpoints[provider_name]
            for model in models:
                is_duplicate = model_id_counts.get(model.id, 1) > 1
                display_id = f"{provider_name}/{model.id}" if is_duplicate else model.id

                new_available_models.append(display_id)
                new_model_context_lengths[display_id] = getattr(
                    model, "context_length", None
                )
                new_model_details[display_id] = {
                    "provider": provider_name,
                    "original_id": model.id,
                    "api_key": provider_config["api_key"],
                    "api_base": provider_config["api_base"],
                }

        self.available_models = sorted(list(set(new_available_models)))
        self.model_context_lengths = new_model_context_lengths
        self.model_details = new_model_details

        self.call_from_thread(
            self.query_one("#chat-history").mount,
            Static(
                f"Found {len(self.available_models)} available models from {len(all_models_by_provider)} provider(s).",
                classes="info-message",
            ),
        )

    def on_mount(self) -> None:
        """Called when the app is mounted. Sets up the agent and MCP connections."""
        # Register and set the custom theme
        self.register_theme(cyberpunk_theme)
        self.theme = "cyberpunk"

        # provider = trace.get_tracer_provider()
        # provider.add_span_processor(SimpleSpanProcessor(self.TextualSpanExporter(self)))
        chat_history = self.query_one("#chat-history")
        chat_history.mount(
            Static(
                "Initializing agent and connecting to tool servers...",
                classes="info-message",
            )
        )
        self.favorite_models = self.settings.get("favoriteModels", [])
        self.run_worker(self.fetch_models, thread=True)

        self.is_loading = True
        self.query_one(Input).placeholder = "Initializing agent..."
        self.run_worker(
            partial(self.setup_agent_and_tools, self.settings),
            name="setup_agent",
            thread=True,
        )
        chat_history.mount(LoadingIndicator(classes="agent-thinking"))
        chat_history.scroll_end()

    def on_unmount(self) -> None:
        """Called when the app is unmounted. Disconnects the MCP client."""
        if self.mcp_clients:
            print("Disconnecting from MCP servers...")
            for client in self.mcp_clients:
                client.disconnect()

        # Close MCP server log files
        for log_file in self.mcp_log_files.values():
            try:
                log_file.close()
            except:
                pass

    def setup_agent_and_tools(self, settings):
        """Initializes the MCP client and the agent with all available tools."""
        logging.info("Starting agent and tool setup...")

        class CaptureIO(io.StringIO):
            encoding = "utf-8"

        @tool
        def aider_edit_file(file: str, instruction: str) -> str:
            """
            Edits a file using the aider tool with a given instruction.
            Args:
                file: The path to the file to edit.
                instruction: The instruction message to pass to aider for the edit.
            """
            output_capture = CaptureIO()
            try:
                # Get the provider details for the current model
                model_details = self.model_details.get(self.model_id)
                if not model_details:
                    return f"Error: Could not find model details for {self.model_id}. Cannot run aider."

                with redirect_stdout(output_capture):
                    # Import aider dependencies here to avoid circular dependencies and keep startup fast
                    from aider.coders import Coder
                    from aider.models import Model

                    # Setup model and coder
                    model = Model(model_details["original_id"])

                    # Add our api key and base to the model's extra_params
                    if not model.extra_params:
                        model.extra_params = {}
                    model.extra_params.update(
                        {
                            "api_key": model_details["api_key"],
                            "api_base": model_details["api_base"],
                            "custom_llm_provider": "openai",
                        }
                    )

                    coder = Coder.create(main_model=model, fnames=[file])

                    # Run the instruction
                    coder.run(instruction)

                result_output = output_capture.getvalue()
                return f"Aider edit log for {file}:\n{result_output}"
            except Exception as e:
                error_output = output_capture.getvalue()
                return f"Error using aider to edit {file}: {e}\nCaptured output:\n{error_output}"

        local_tools = [aider_edit_file]
        self.tools_by_source = {"inbuilt": local_tools}
        mcp_tools = []
        self.settings = settings
        server_configs = self.settings.get("mcpServers", {})
        logging.info(f"Found {len(server_configs)} MCP server configurations.")

        # Create logs directory structure
        mcp_log_dir = self.log_dir / "mcp"
        mcp_log_dir.mkdir(exist_ok=True)

        if server_configs:
            try:
                # Monkey patch stdio_client to use our custom log files
                from mcp.client.stdio import stdio_client

                original_stdio_client = stdio_client

                def patched_stdio_client(server, errlog=None):
                    # Find the matching log file for this server
                    server_key = f"{server.command} {' '.join(server.args)}"
                    custom_errlog = self.mcp_log_files.get(server_key)

                    if custom_errlog:
                        # Use our custom log file
                        return original_stdio_client(server, custom_errlog)
                    else:
                        # Fall back to default behavior
                        return original_stdio_client(server, errlog)

                # Apply the monkey patch
                import mcp.client.stdio

                mcp.client.stdio.stdio_client = patched_stdio_client

                # Also patch the import in mcpadapt if it exists
                try:
                    import mcpadapt.core

                    mcpadapt.core.stdio_client = patched_stdio_client
                except ImportError:
                    pass

                for name, config in server_configs.items():
                    logging.info(f"Processing MCP server config: {name}")
                    # Skip disabled servers
                    if config.get("disabled", False):
                        logging.info(f"Skipping disabled MCP server: {name}")
                        continue

                    command = config.get("command")
                    args = config.get("args", [])
                    env = config.get("env", {})
                    if command:
                        full_env = {**os.environ, **env}

                        # Create log file for this server's stderr
                        log_file_path = mcp_log_dir / f"{name}.log"
                        log_file = open(log_file_path, "w")

                        # Create a unique key for this server (command + args)
                        server_key = f"{command} {' '.join(args)}"
                        self.mcp_log_files[server_key] = log_file

                        logging.info(f"[{name}] Creating StdioServerParameters...")
                        server_param = StdioServerParameters(
                            command=command, args=args, env=full_env
                        )
                        logging.info(f"[{name}] Creating MCPClient...")
                        client = MCPClient([server_param])
                        self.mcp_clients.append(client)

                        logging.info(f"[{name}] Getting tools from server...")
                        server_tools = client.get_tools()
                        logging.info(f"[{name}] Found {len(server_tools)} tools.")

                        self.tools_by_source[name] = server_tools
                        mcp_tools.extend(server_tools)

                # Restore original function
                mcp.client.stdio.stdio_client = original_stdio_client
                if "mcpadapt.core" in sys.modules:
                    mcpadapt.core.stdio_client = original_stdio_client

                logging.info(
                    f"Successfully connected to {len(mcp_tools)} tools from MCP servers."
                )
            except Exception as e:
                # Restore original function in case of error
                try:
                    mcp.client.stdio.stdio_client = original_stdio_client
                    if "mcpadapt.core" in sys.modules:
                        mcpadapt.core.stdio_client = original_stdio_client
                except:
                    pass

                self.call_from_thread(
                    self.query_one("#chat-history").mount,
                    Static(
                        f"Error connecting to MCP servers: {e}", classes="error-message"
                    ),
                )

                # Close log files on error
                for log_file in self.mcp_log_files.values():
                    try:
                        log_file.close()
                    except:
                        pass
                self.mcp_log_files.clear()

        self.all_tools = local_tools + mcp_tools
        logging.info(f"Total tools initialized: {len(self.all_tools)}")

        logging.info("Updating agent with new model...")
        self.update_agent_with_new_model(self.model_id, startup=True)
        logging.info("Agent and tool setup complete.")

    def update_agent_with_new_model(self, model_id: str, startup: bool = False):
        """Creates a new agent with the specified model."""
        if model_id not in self.model_details:
            if not startup:
                self.query_one("#chat-history").mount(
                    Static(
                        f"Error: Details for model '{model_id}' not found.",
                        classes="error-message",
                    )
                )
            # Don't proceed if we don't have details. fetch_models should run first.
            return

        self.model_id = model_id
        context_length = self.model_context_lengths.get(model_id)
        self.max_context_length = (
            context_length if context_length is not None else self.global_context_length
        )
        self.context_threshold = int(self.max_context_length * 0.8)

        # Get provider-specific details for the selected model
        details = self.model_details[model_id]

        # Create a dedicated OpenAI Server model object using instance variables
        openai_server_model = OpenAIServerModel(
            model_id=details["original_id"],
            api_key=details["api_key"],
            api_base=details["api_base"],
        )

        # Pass the model object to the agent
        self.agent = ToolCallingAgent(
            model=openai_server_model,
            tools=self.all_tools,
            add_base_tools=True,
        )

        self.query_one(Header).title = f"Vibe Agent ({self.model_id})"

        if not startup:
            chat_history = self.query_one("#chat-history")
            chat_history.mount(
                Static(f"Switched to model: {self.model_id}", classes="info-message")
            )
            logging.info(f"Model switched to: {self.model_id}")

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when a worker's state changes."""
        if event.worker.name == "setup_agent":
            # Remove the loading indicator when the setup worker finishes
            if event.worker.state == WorkerState.SUCCESS:
                # The worker is done, update the UI.
                self.is_loading = False
                logging.info(f"Model initialized: {self.model_id}")
                self.query_one("#chat-history").mount(
                    Static(
                        f"Agent is ready. Model: {self.model_id}. Type your message and press Enter.",
                        classes="info-message",
                    )
                )
                try:
                    self.query_one(".agent-thinking").remove()
                except Exception:
                    pass  # It might have already been removed
                self.query_one(Input).placeholder = "Ask the agent to do something..."
                self.query_one("#input").focus(scroll_visible=True)
            elif event.worker.state == WorkerState.ERROR:
                # The worker failed
                self.is_loading = False
                logging.error(f"Agent setup failed: {event.worker.error}")
                self.query_one("#chat-history").mount(
                    Static(
                        "Agent initialization failed. Check logs.",
                        classes="error-message",
                    )
                )
                try:
                    self.query_one(".agent-thinking").remove()
                except Exception:
                    pass  # It might have already been removed
                self.query_one(Input).placeholder = "Agent failed to initialize."

    def compose(self) -> ComposeResult:
        """Creates the layout for the chat application."""
        yield Header()
        yield ScrollableContainer(id="chat-history")
        input_widget = Input(id="input", placeholder="Ask the agent to do something...")
        yield input_widget
        yield AutoComplete(input_widget, candidates=self.get_autocomplete_candidates)
        yield Footer()

    def get_agent_response(self, user_message: str) -> None:
        """Worker to get response from agent."""
        self.call_from_thread(self._update_telemetry_status)
        if self.agent:
            try:
                allowed_paths = self.settings.get("allowedPaths", [])
                current_dir = os.getcwd()
                context_dict = {
                    "allowedPaths": ", ".join(allowed_paths),
                    "pwd": current_dir,
                }
                response = self.agent.run(
                    user_message, additional_args=context_dict, reset=False
                )
                self.post_message(self.AgentResponse(response))
            except Exception as e:
                self.post_message(self.AgentResponse(f"Error: {str(e)}"))
                logging.error(f"Agent response error: {e}", exc_info=True)

    def on_chat_app_agent_response(self, message: AgentResponse) -> None:
        """Handles the agent's response."""
        # If the worker is None, it means the job was cancelled.
        if self.agent_worker is None:
            return

        self.is_loading = False
        self.agent_worker = None
        self.query_one(Input).placeholder = "Ask the agent to do something..."

        chat_history = self.query_one("#chat-history")
        chat_history.query(".agent-thinking").last().remove()

        response_text = message.response
        if self.glitch_mode:
            response_text = self._apply_glitch(response_text)

        chat_history.mount(Markdown(response_text, classes="agent-message"))
        chat_history.scroll_end()

    def get_autocomplete_candidates(self, state: TargetState) -> list[DropdownItem]:
        """Provides dynamic candidates for the autocomplete dropdown."""
        text = state.text.lstrip()

        if not text.startswith("/"):
            return []  # No suggestions if it's not a command

        if " " in text:
            command, _ = text.split(" ", 1)
            if command == "/model":
                # User is typing a model name after /model
                sorted_models = sorted(
                    self.available_models,
                    key=lambda model_id: (
                        model_id not in self.favorite_models,
                        model_id,
                    ),
                )
                return [
                    DropdownItem(f"/model {model_id}") for model_id in sorted_models
                ]
            elif command == "/compress":
                strategies = ["drop_oldest", "middle_out", "summarize"]
                return [
                    DropdownItem(f"/compress {strategy}") for strategy in strategies
                ]
            else:
                return []  # No suggestions for arguments of other commands for now
        else:
            # User is typing a command, textual-autocomplete will filter
            commands = [
                "/quit",
                "/tools",
                "/model",
                "/refresh-models",
                "/compress",
                "/dump-context",
                "/show-settings",
            ]
            return [DropdownItem(cmd) for cmd in commands]

    def on_chat_app_tool_call(self, message: ToolCall) -> None:
        """Handles a tool call message."""
        chat_history = self.query_one("#chat-history")
        chat_history.mount(
            Static(f"Calling tool: {message.tool_name}", classes="tool-call-message")
        )
        chat_history.scroll_end()

    def _handle_command(self, user_message: str) -> bool:
        """
        Handles a user command.
        Returns True if a command was handled, False otherwise.
        """
        parts = user_message.strip().split(" ", 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        chat_history = self.query_one("#chat-history")

        if command == "/quit":
            self.exit()
            return True

        if command == "/xyzzy":
            strength_str = arg.strip() if arg else None
            message = ""

            if strength_str:
                # Argument provided: set strength and ensure mode is on.
                try:
                    strength = float(strength_str)
                    if 0 <= strength <= 100:
                        self.glitch_mode = True
                        self.glitch_strength = strength / 100.0
                        message = f"Glitch mode enabled. Strength set to {strength}%."
                    else:
                        message = "Glitch strength must be between 0 and 100."
                except ValueError:
                    message = f"Invalid strength value: '{strength_str}'. Please provide a number."
            else:
                # No argument: toggle mode.
                self.glitch_mode = not self.glitch_mode
                message = (
                    f"Glitch mode {'enabled' if self.glitch_mode else 'disabled'}."
                )

            # chat_history.mount(Static(message, classes="info-message"))
            # chat_history.scroll_end()
            return True

        if command == "/tools":
            self.list_tools()
            return True

        if command == "/refresh-models":
            chat_history.mount(
                Static("Refreshing model list from API...", classes="info-message")
            )
            chat_history.scroll_end()
            self.run_worker(self.fetch_models, thread=True)
            return True

        if command == "/model":
            if arg:
                self.update_model(arg.strip())
            else:
                self.action_select_model()
            return True
        if command == "/compress":
            strategy = arg.strip() if arg else self.default_strategy
            self.compress_context(strategy)
            return True
        if command == "/dump-context":
            self.dump_context(arg.strip() if arg else "markdown")
            return True
        if command == "/show-settings":
            self.show_settings()
            return True
        return False

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles user input and displays the agent's response."""
        if self.is_loading:
            return

        user_message = event.value.strip()

        if not user_message:
            return

        chat_history = self.query_one("#chat-history")
        chat_history.mount(Static(f"> {user_message}", classes="user-message"))
        self.query_one(Input).value = ""
        self.command_history.append(user_message)
        self.history_index = len(self.command_history)
        chat_history.scroll_end()

        if user_message.startswith("/") and self._handle_command(user_message):
            return

        if not self.agent:
            chat_history.mount(
                Static(
                    "Agent is not yet initialized. Please wait.",
                    classes="error-message",
                )
            )
            chat_history.scroll_end()
            return

        self.is_loading = True
        self.query_one(Input).placeholder = "Waiting for response..."

        chat_history.mount(LoadingIndicator(classes="agent-thinking"))
        chat_history.scroll_end()

        self.agent_worker = self.run_worker(
            partial(self.get_agent_response, user_message),
            thread=True,
            name=f"Agent request: {user_message[:30]}",
        )

    def update_model(self, new_model_id: str):
        """Callback to update the model."""
        if self.available_models and new_model_id in self.available_models:
            self.update_agent_with_new_model(new_model_id)
        else:
            # Check if the model exists on the server
            self.run_worker(
                partial(self.verify_and_update_model, new_model_id), thread=True
            )

    async def verify_and_update_model(self, model_id: str):
        """Verify model exists and then update."""
        from openai import AsyncOpenAI

        # This command is now more complex. A user might type `/model my-model`
        # or `/model provider/my-model`.
        # For now, we will rely on the periodic `fetch_models` to discover models.
        # This function will just check if the model is in our available list.

        chat_history = self.query_one("#chat-history")
        if model_id in self.available_models:
            self.call_from_thread(self.update_agent_with_new_model, model_id)
        else:
            # Try to refresh models to see if it's new
            await self.fetch_models()
            if model_id in self.available_models:
                self.call_from_thread(self.update_agent_with_new_model, model_id)
            else:
                self.call_from_thread(
                    chat_history.mount,
                    Static(
                        f"Model '{model_id}' not found after refreshing. Please check the name or provider.",
                        classes="error-message",
                    ),
                )

    def list_tools(self) -> None:
        """Displays the list of available tools."""
        chat_history = self.query_one("#chat-history")
        if not self.agent or not hasattr(self.agent, "tools"):
            chat_history.mount(
                Static(
                    "No tools available or agent not initialized.",
                    classes="error-message",
                )
            )
            chat_history.scroll_end()
            return

        tool_items = []

        # MCP and local tools from self.tools_by_source
        for source, tools in self.tools_by_source.items():
            if tools:
                tool_items.append(
                    f"\n[bold underline]{source.capitalize()} Tools[/bold underline]"
                )
                sorted_tools = sorted(tools, key=lambda t: t.name)
                for tool in sorted_tools:
                    description = getattr(
                        tool, "description", "No description available."
                    )
                    first_line = description.strip().split("\n")[0]
                    tool_items.append(f"• [bold]{tool.name}[/bold]: {first_line}")

        # Base tools
        agent_tool_names = set(self.agent.tools.keys())
        all_my_tool_names = {t.name for t in self.all_tools}
        base_tool_names = agent_tool_names - all_my_tool_names

        if base_tool_names:
            tool_items.append("\n[bold underline]Base Tools[/bold underline]")
            for tool_name in sorted(list(base_tool_names)):
                tool = self.agent.tools[tool_name]
                description = getattr(tool, "description", "No description available.")
                first_line = description.strip().split("\n")[0]
                tool_items.append(f"• [bold]{tool.name}[/bold]: {first_line}")

        if not tool_items:
            tool_list_str = "No tools available."
        else:
            # remove first newline if present
            if tool_items and tool_items[0].startswith("\n"):
                tool_items[0] = tool_items[0][1:]
            tool_list_str = "\n".join(tool_items)

        chat_history.mount(Static(tool_list_str, classes="info-message"))
        chat_history.scroll_end()

    def action_history_prev(self) -> None:
        """Go to the previous command in history."""
        if self.history_index > 0:
            self.history_index -= 1
            self.query_one(Input).value = self.command_history[self.history_index]

    def action_history_next(self) -> None:
        """Go to the next command in history."""
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.query_one(Input).value = self.command_history[self.history_index]
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            self.query_one(Input).value = ""

    def action_scroll_up(self) -> None:
        """Scroll the chat history up."""
        self.query_one("#chat-history").scroll_page_up()

    def action_scroll_down(self) -> None:
        """Scroll the chat history down."""
        self.query_one("#chat-history").scroll_page_down()

    def action_select_model(self) -> None:
        """Show the model selection screen."""
        if not self.available_models:
            self.query_one("#chat-history").mount(
                Static(
                    "Model list not available yet. Please try again shortly.",
                    classes="error-message",
                )
            )
            self.query_one("#chat-history").scroll_end()
            return

        def on_model_selected(model_id: str):
            if model_id:
                self.update_model(model_id)

        self.push_screen(
            ModelSelectScreen(
                self.available_models,
                self.favorite_models,
                self.model_id,
                self.model_details,
            ),
            on_model_selected,
        )

    def action_cancel_request(self) -> None:
        """Cancels the current in-progress agent request."""
        if self.is_loading and self.agent_worker:
            self.agent_worker.cancel()
            self.is_loading = False
            self.agent_worker = None

            chat_history = self.query_one("#chat-history")
            # Remove the thinking indicator
            try:
                chat_history.query(".agent-thinking").last().remove()
            except Exception:
                pass  # It might have already been removed

            chat_history.mount(Static("Request cancelled.", classes="info-message"))
            self.query_one(Input).placeholder = "Ask the agent to do something..."
            chat_history.scroll_end()

    def get_current_context_tokens(self) -> int:
        """Calculates the total token count from the agent's memory."""
        if not self.agent or not self.agent.memory.steps:
            return 0

        total_tokens = 0
        for step in self.agent.memory.steps:
            if hasattr(step, "token_usage") and step.token_usage:
                total_tokens += step.token_usage.total_tokens
        return total_tokens

    def _format_context_as_markdown(self, messages: list[dict]) -> str:
        """Formats the agent's message history into a readable markdown string."""
        markdown_lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            markdown_lines.append(f"### {role.replace('_', ' ').title()}")
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_content = part.get("text", "")
                        markdown_lines.append(text_content)
                    elif part.get("type") == "image":
                        markdown_lines.append("*[Image Content]*")
            elif isinstance(content, str):
                markdown_lines.append(content)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                markdown_lines.append("\n**Tool Calls:**")
                for tc in tool_calls:
                    function = tc.get("function", {})
                    name = function.get("name", "N/A")
                    arguments = function.get("arguments", "{}")
                    markdown_lines.append(f"- **{name}**")
                    try:
                        args_dict = (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else arguments
                        )
                        pretty_args = json.dumps(args_dict, indent=2)
                        markdown_lines.append(f"  ```json\n{pretty_args}\n  ```")
                    except (json.JSONDecodeError, TypeError):
                        markdown_lines.append(f"  Arguments: `{arguments}`")
            token_usage = msg.get("token_usage")
            if token_usage and token_usage.get("total_tokens", 0) > 0:
                markdown_lines.append(f"\n*Tokens: {token_usage.get('total_tokens')}*")
            markdown_lines.append("\n---")
        return "\n".join(markdown_lines)

    def dump_context(self, output_format: str = "markdown"):
        """Dumps the current context to the UI."""
        if not self.agent:
            self.query_one("#chat-history").mount(
                Static("Agent is not initialized.", classes="error-message")
            )
            return

        chat_history = self.query_one("#chat-history")
        try:
            messages = self.agent.write_memory_to_messages()
            messages_as_dicts = [m.dict() for m in messages]
            for m in messages_as_dicts:
                if "raw" in m:
                    del m["raw"]

            if output_format == "json":
                json_context = json.dumps(messages_as_dicts, indent=2)
                markdown_content = f"```json\n{json_context}\n```"
            else:
                markdown_content = self._format_context_as_markdown(messages_as_dicts)

            chat_history.mount(Markdown(markdown_content, classes="info-message"))

        except Exception as e:
            chat_history.mount(
                Static(f"Error dumping context: {e}", classes="error-message")
            )
        finally:
            chat_history.scroll_end()

    def show_settings(self):
        """Displays configuration and log paths, and settings content."""
        chat_history = self.query_one("#chat-history")
        try:
            settings_path = self.config_dir / "settings.json"
            if settings_path.exists():
                with open(settings_path, "r") as f:
                    settings_content = json.load(f)
                settings_dump = json.dumps(settings_content, indent=2)
            else:
                settings_dump = "settings.json not found."

            info_text = (
                f"**Config Path:** {self.config_dir}\n\n"
                f"**Logs Path:**   {self.log_dir}\n\n"
                f"**settings.json:**\n\n"
                f"```json\n{settings_dump}\n```"
            )

            chat_history.mount(Markdown(info_text, classes="info-message"))

        except Exception as e:
            chat_history.mount(
                Static(f"Error showing settings: {e}", classes="error-message")
            )
        finally:
            chat_history.scroll_end()

    def compress_context(self, strategy: str):
        if not self.agent:
            self.query_one("#chat-history").mount(
                Static("Agent is not initialized.", classes="error-message")
            )
            return

        if strategy not in ["drop_oldest", "middle_out", "summarize"]:
            self.query_one("#chat-history").mount(
                Static(f"Unknown strategy: {strategy}", classes="error-message")
            )
            return

        initial_tokens = self.get_current_context_tokens()

        # if current_tokens <= self.context_threshold:
        #     self.query_one("#chat-history").mount(
        #         Static("Context is already under limit.", classes="info-message")
        #     )
        #     return

        chat_history = self.query_one("#chat-history")

        # Display status message
        if strategy == "summarize":
            status_text = "Summarizing and compressing context..."
        else:
            status_text = f"Compressing context using {strategy}..."

        status_widget = Static(status_text, classes="agent-thinking")
        chat_history.mount(status_widget)
        chat_history.scroll_end()

        # Perform compression
        if strategy == "drop_oldest":
            current_tokens = initial_tokens
            while (
                current_tokens > self.context_threshold
                and len(self.agent.memory.steps) > 1
            ):
                self.agent.memory.steps.pop(0)
                current_tokens = self.get_current_context_tokens()
        elif strategy == "middle_out":
            current_tokens = initial_tokens
            steps = self.agent.memory.steps
            while current_tokens > self.context_threshold and len(steps) > 2:
                middle = len(steps) // 2
                del steps[middle]
                current_tokens = self.get_current_context_tokens()
        elif strategy == "summarize":
            messages = self.agent.write_memory_to_messages()
            history_text = "\n".join(
                [
                    f"{m.role.value}: {''.join(c['text'] if isinstance(c, dict) and 'text' in c else str(c) for c in (m.content if isinstance(m.content, list) else [m.content]))}"
                    for m in messages
                ]
            )
            summary_prompt = f"""
            Summarize the following conversation history concisely while retaining key information. 
            DON'T summarize the tools available, but do include a summary of the tools used. 
            ALWAYS include the actual content of the final answer.
            If the user has been generating code or files, include the final version of the code 
            or final file path(s):\n\n{history_text}
            """.strip()

            # Create a dedicated summarization agent
            summary_model_details = self.model_details[self.model_id]
            summarizer_model = OpenAIServerModel(
                model_id=summary_model_details["original_id"],
                api_key=summary_model_details["api_key"],
                api_base=summary_model_details["api_base"],
            )
            summarizer_agent = ToolCallingAgent(
                model=summarizer_model,
                tools=[],  # No special tools for now
                add_base_tools=True,  # Needs final_answer tool
            )

            # Run the summarizer agent to get the summary
            summary = summarizer_agent.run(summary_prompt)

            # Get the token usage from the summarizer agent's last step
            summarizer_token_usage = None
            if summarizer_agent.memory.steps:
                last_step = summarizer_agent.memory.steps[-1]
                if hasattr(last_step, "token_usage") and last_step.token_usage:
                    summarizer_token_usage = last_step.token_usage

            # Create a simple summary step to replace the history
            timing = Timing(start_time=time.time())
            timing.end_time = time.time()
            summary_step = ActionStep(
                step_number=1,
                timing=timing,
                model_output="Conversation summary:",
                observations=summary,
                token_usage=summarizer_token_usage
                or TokenUsage(
                    input_tokens=0, output_tokens=0
                ),  # Use summarizer's token usage or fallback
            )

            # Replace the agent's memory with just the summary
            self.agent.memory.steps = [summary_step]
            self.agent.step_number = 2

        # Remove status message
        status_widget.remove()

        # Display final result
        new_tokens = self.get_current_context_tokens()
        chat_history.mount(
            Static(
                f"Context compressed from {initial_tokens} to {new_tokens} tokens.",
                classes="info-message",
            )
        )
        chat_history.scroll_end()

    def _apply_glitch(self, text: str) -> str:
        """Corrupts text with Zalgo and other glitches."""
        glitched_chars = []
        for char in text:
            # Don't glitch whitespace
            if not char.strip():
                glitched_chars.append(char)
                continue

            # Roll for a glitch
            if random.random() < self.glitch_strength:
                glitch_type = random.choice(
                    ["delete", "substitute", "replace_high", "zalgo", "leetspeak"]
                )

                if glitch_type == "delete":
                    continue  # Skip appending

                elif glitch_type == "substitute":
                    glitched_chars.append(
                        random.choice(
                            string.ascii_letters + string.digits + string.punctuation
                        )
                    )

                elif glitch_type == "replace_high":
                    glitched_chars.append(random.choice(self.HIGH_ASCII_CHARS))

                elif glitch_type == "leetspeak":
                    glitched_chars.append(self.LEETSPEAK_MAP.get(char, char))

                elif glitch_type == "zalgo":
                    # Start with the original character
                    zalgoed_char = char
                    # Determine intensity
                    max_diacritics = 1 + int(self.glitch_strength * 30)

                    # Add UP diacritics
                    for _ in range(random.randint(0, max_diacritics)):
                        zalgoed_char += random.choice(self.ZALGO_UP)
                    # Add MID diacritics
                    for _ in range(random.randint(0, max_diacritics // 2)):
                        zalgoed_char += random.choice(self.ZALGO_MID)
                    # Add DOWN diacritics
                    for _ in range(random.randint(0, max_diacritics)):
                        zalgoed_char += random.choice(self.ZALGO_DOWN)

                    glitched_chars.append(zalgoed_char)

            else:  # No glitch
                glitched_chars.append(char)

        return "".join(glitched_chars)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VibeAgent - A smolagents-based chat interface"
    )
    parser.add_argument(
        "--model", help="Override the default model to use (from settings.json)"
    )
    # --api-key-env-var and --api-base are removed as they are now endpoint-specific
    args = parser.parse_args()

    APP_NAME = "vibeagent"
    config_dir = Path(platformdirs.user_config_dir(APP_NAME))
    log_dir = Path(platformdirs.user_log_dir(APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load settings to get model configuration
    def load_settings(config_dir: Path):
        """Load settings from settings.json with environment variable substitution."""
        settings_path = config_dir / "settings.json"
        if not settings_path.exists():
            print(
                f"settings.json not found in {config_dir}, creating from default-settings.json..."
            )
            if os.path.exists("default-settings.json"):
                try:
                    import shutil

                    shutil.copyfile("default-settings.json", settings_path)
                except IOError as e:
                    print(
                        f"Error copying from default-settings.json: {e}. Creating empty settings file."
                    )
                    with open(settings_path, "w") as f_settings:
                        json.dump({"mcpServers": {}}, f_settings, indent=2)
            else:
                print(
                    "Warning: default-settings.json not found. Creating empty settings.json."
                )
                with open(settings_path, "w") as f_settings:
                    json.dump({"mcpServers": {}}, f_settings, indent=2)

        try:
            with open(settings_path, "r") as f:
                config = json.load(f)

                def _substitute_env_vars(data):
                    if isinstance(data, dict):
                        return {k: _substitute_env_vars(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [_substitute_env_vars(i) for i in data]
                    elif isinstance(data, str):
                        return os.path.expandvars(data)
                    else:
                        return data

                config = _substitute_env_vars(config)
                if isinstance(config, dict):
                    return config
                else:
                    print(
                        "Warning: settings.json does not contain a valid JSON object. Using empty config."
                    )
                    return {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading settings.json: {e}. Using empty config.")
            return {}

    # Load settings and determine model configuration
    settings = load_settings(config_dir)
    model_config = settings

    # Determine the initial model to use
    initial_model_id = args.model or model_config.get("defaultModel")

    if not initial_model_id:
        print(
            "Error: No default model specified in settings.json and --model not provided."
        )
        sys.exit(1)

    if not model_config.get("endpoints"):
        print("Error: No model endpoints configured in settings.json.")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_dir / "vibeagent.log",
        filemode="w",
    )
    stdout_logger = logging.getLogger("STDOUT")
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    stderr_logger = logging.getLogger("STDERR")
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    print("--- VibeAgent session started, logging to vibeagent.log ---")

    app = ChatApp(
        model_config=model_config,
        initial_model_id=initial_model_id,
        config_dir=config_dir,
        log_dir=log_dir,
    )
    app.settings = settings  # Set the settings after instantiation
    app.run()
