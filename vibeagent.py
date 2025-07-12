# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "smolagents[litellm,mcp,telemetry,toolkit]",
#   "textual",
#   "python-dotenv",
#   "aider-chat",
#   "openai",
#   "textual-autocomplete",
# ]
# ///

import logging
import sys
import json
import os
import io
import argparse
import time
from functools import partial
from dotenv import load_dotenv
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

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# Load environment variables from .env file
load_dotenv()


@tool
def aider_edit_file(file: str, instruction: str) -> str:
    """
    Edits a file using the aider tool with a given instruction.
    Args:
        file: The path to the file to edit.
        instruction: The instruction message to pass to aider for the edit.
    """
    try:
        # Import aider dependencies here to avoid circular dependencies and keep startup fast
        from aider.client import AiderClient
        from aider.io import IO

        # Capture aider's output
        output_capture = io.StringIO()

        def capture_output(text, **kwargs):
            output_capture.write(text)
            if not text.endswith("\n"):
                output_capture.write("\n")

        # Setup custom IO for aider
        aider_io = IO(
            user_output_callback=capture_output,
            confirm_ask=lambda: True,  # Auto-confirm any prompts
        )

        # Make sure API keys are loaded for aider
        load_dotenv()
        client = AiderClient(fnames=[file], yes=True, io=aider_io)
        client.run_chat(with_message=instruction)

        result_output = output_capture.getvalue()
        output_capture.close()

        return f"Aider edit log for {file}:\n{result_output}"
    except ImportError:
        return (
            "Error: 'aider-chat' is not installed. Please install it to use this tool."
        )
    except Exception as e:
        return f"Error using aider to edit {file}: {e}"


class ModelSelectScreen(ModalScreen[str]):
    """Screen for selecting a model."""

    def __init__(
        self, models: list[str], favorites: list[str], current_model: str
    ) -> None:
        super().__init__()
        self.all_models = models
        self.favorites = favorites
        self.current_model = current_model

    def compose(self) -> ComposeResult:
        # separate favorite models and other models
        fav_options = []
        other_options = []

        # Create options, favorites first
        for model in self.all_models:
            if model in self.favorites:
                fav_options.append(Option(f"{model} (favorite)", id=model))
            else:
                other_options.append(Option(model, id=model))

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
    }

    .error-message {
        background: $panel;
        color: $error;
        padding: 1;
        margin-bottom: 1;
        border-left: thick $error;
        text-wrap: wrap;
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
    ]
    COMMANDS = App.COMMANDS | {ModelSelectProvider}

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
        model_id: str,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__()
        self.title = "Vibe Agent Chat"
        self.agent = None
        self.mcp_clients = []
        self.command_history = []
        self.history_index = -1
        self.settings = {}
        self.mcp_log_files = {}  # Store MCP server log files
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
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
        self.model_context_lengths = {}  # Initialize model_context_lengths
        self.is_loading = False

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
        """Fetches available models from the OpenAI-compatible endpoint."""
        try:
            from openai import AsyncOpenAI

            async with AsyncOpenAI(
                api_key=self.api_key, base_url=self.api_base
            ) as client:
                models = await client.models.list()
                self.available_models = sorted([model.id for model in models.data])
                self.model_context_lengths = {
                    model.id: getattr(model, "context_length", None)
                    for model in models.data
                }
                self.call_from_thread(
                    self.query_one("#chat-history").mount,
                    Static(
                        f"Found {len(self.available_models)} available models.",
                        classes="info-message",
                    ),
                )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#chat-history").mount,
                Static(f"Error fetching models: {e}", classes="error-message"),
            )

    def on_mount(self) -> None:
        """Called when the app is mounted. Sets up the agent and MCP connections."""
        # Register and set the custom theme
        self.register_theme(cyberpunk_theme)
        self.theme = "cyberpunk"

        # provider = trace.get_tracer_provider()
        # provider.add_span_processor(SimpleSpanProcessor(self.TextualSpanExporter(self)))

        self.query_one("#chat-history").mount(
            Static(
                "Initializing agent and connecting to tool servers...",
                classes="info-message",
            )
        )
        self.favorite_models = self.settings.get("favoriteModels", [])
        self.run_worker(self.fetch_models, thread=True)

        self.setup_agent_and_tools(self.settings)

        # Log the model name during initialization
        logging.info(f"Model initialized: {self.model_id}")

        self.query_one("#chat-history").mount(
            Static(
                f"Agent is ready. Model: {self.model_id}. Type your message and press Enter.",
                classes="info-message",
            )
        )
        self.query_one("#input").focus(scroll_visible=True)

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
        local_tools = [aider_edit_file]
        self.tools_by_source = {"inbuilt": local_tools}
        mcp_tools = []
        self.settings = settings
        server_configs = self.settings.get("mcpServers", {})

        # Create logs directory structure
        logs_dir = os.path.join("logs", "mcp")
        os.makedirs(logs_dir, exist_ok=True)

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
                    # Skip disabled servers
                    if config.get("disabled", False):
                        print(f"Skipping disabled MCP server: {name}")
                        continue

                    command = config.get("command")
                    args = config.get("args", [])
                    env = config.get("env", {})
                    if command:
                        full_env = {**os.environ, **env}

                        # Create log file for this server's stderr
                        log_file_path = os.path.join(logs_dir, f"{name}.log")
                        log_file = open(log_file_path, "w")

                        # Create a unique key for this server (command + args)
                        server_key = f"{command} {' '.join(args)}"
                        self.mcp_log_files[server_key] = log_file

                        server_param = StdioServerParameters(
                            command=command, args=args, env=full_env
                        )
                        client = MCPClient([server_param])
                        self.mcp_clients.append(client)
                        server_tools = client.get_tools()
                        self.tools_by_source[name] = server_tools
                        mcp_tools.extend(server_tools)
                        print(f"Found {len(server_tools)} tools from '{name}' server.")

                # Restore original function
                mcp.client.stdio.stdio_client = original_stdio_client
                if "mcpadapt.core" in sys.modules:
                    mcpadapt.core.stdio_client = original_stdio_client

                print(
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

                self.query_one("#chat-history").mount(
                    Static(
                        f"Error connecting to MCP servers: {e}", classes="error-message"
                    )
                )

                # Close log files on error
                for log_file in self.mcp_log_files.values():
                    try:
                        log_file.close()
                    except:
                        pass
                self.mcp_log_files.clear()

        self.all_tools = local_tools + mcp_tools

        self.update_agent_with_new_model(self.model_id, startup=True)

    def update_agent_with_new_model(self, model_id: str, startup: bool = False):
        """Creates a new agent with the specified model."""
        self.model_id = model_id
        context_length = self.model_context_lengths.get(model_id)
        self.max_context_length = (
            context_length if context_length is not None else self.global_context_length
        )
        self.context_threshold = int(self.max_context_length * 0.8)

        # Create a dedicated OpenAI Server model object using instance variables
        openai_server_model = OpenAIServerModel(
            model_id=self.model_id,
            api_key=self.api_key,
            api_base=self.api_base,
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
        self.is_loading = False
        self.query_one(Input).placeholder = "Ask the agent to do something..."

        chat_history = self.query_one("#chat-history")
        chat_history.query(".agent-thinking").last().remove()
        chat_history.mount(Markdown(message.response, classes="agent-message"))
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
            self.dump_context()
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

        self.run_worker(
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
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
            details = await client.models.retrieve(model_id)
            ctx_len = getattr(details, "context_length", None)
            if ctx_len:
                self.model_context_lengths[model_id] = ctx_len
            # It exists, update it in the main thread
            self.call_from_thread(self.update_agent_with_new_model, model_id)
            if model_id not in self.available_models:
                self.available_models.append(model_id)
                self.available_models.sort()

        except Exception:
            self.call_from_thread(
                self.query_one("#chat-history").mount,
                Static(
                    f"Model '{model_id}' not found or invalid.", classes="error-message"
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
                self.available_models, self.favorite_models, self.model_id
            ),
            on_model_selected,
        )

    def get_current_context_tokens(self) -> int:
        """Calculates the total token count from the agent's memory."""
        if not self.agent or not self.agent.memory.steps:
            return 0

        total_tokens = 0
        for step in self.agent.memory.steps:
            if hasattr(step, "token_usage") and step.token_usage:
                total_tokens += step.token_usage.total_tokens
        return total_tokens

    def dump_context(self):
        """Dumps the current context to the UI as JSON."""
        if not self.agent:
            self.query_one("#chat-history").mount(
                Static("Agent is not initialized.", classes="error-message")
            )
            return

        chat_history = self.query_one("#chat-history")
        try:
            messages = self.agent.write_memory_to_messages()
            # Convert messages to a list of dictionaries for JSON serialization
            messages_as_dicts = [m.dict() for m in messages]
            # clean up raw field
            for m in messages_as_dicts:
                if "raw" in m:
                    del m["raw"]

            json_context = json.dumps(messages_as_dicts, indent=2)

            # Create a Markdown block for the JSON
            markdown_content = f"```json\n{json_context}\n```"
            chat_history.mount(Markdown(markdown_content, classes="info-message"))

        except Exception as e:
            chat_history.mount(
                Static(f"Error dumping context: {e}", classes="error-message")
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
            summarizer_model = OpenAIServerModel(
                model_id=self.model_id,
                api_key=self.api_key,
                api_base=self.api_base,
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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VibeAgent - A smolagents-based chat interface"
    )
    parser.add_argument(
        "--model", help="Override the model to use (overrides settings.json and .env)"
    )
    parser.add_argument(
        "--api-key-env-var",
        help="Environment variable name to get API key from (overrides settings.json)",
    )
    parser.add_argument(
        "--api-base",
        help="Override the API base URL to use (overrides settings.json and .env)",
    )
    args = parser.parse_args()

    # Load settings to get model configuration
    def load_settings():
        """Load settings from settings.json with environment variable substitution."""
        if not os.path.exists("settings.json"):
            print("settings.json not found, creating from default-settings.json...")
            if os.path.exists("default-settings.json"):
                try:
                    import shutil

                    shutil.copyfile("default-settings.json", "settings.json")
                except IOError as e:
                    print(
                        f"Error copying from default-settings.json: {e}. Creating empty settings file."
                    )
                    with open("settings.json", "w") as f_settings:
                        json.dump({"mcpServers": {}}, f_settings, indent=2)
            else:
                print(
                    "Warning: default-settings.json not found. Creating empty settings.json."
                )
                with open("settings.json", "w") as f_settings:
                    json.dump({"mcpServers": {}}, f_settings, indent=2)

        try:
            with open("settings.json", "r") as f:
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
    settings = load_settings()
    model_config = settings.get("model", {})

    # Determine model configuration with priority: command line > settings.json > environment variables
    model_id = args.model or model_config.get(
        "id", os.getenv("MODEL", "mistralai/devstral-small:free")
    )

    # Handle API key with priority: command line env var > settings.json > default env vars
    if args.api_key_env_var:
        api_key = os.getenv(args.api_key_env_var)
        if not api_key:
            print(f"Error: Environment variable '{args.api_key_env_var}' is not set.")
            sys.exit(1)
    else:
        api_key = model_config.get(
            "api_key", os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

    api_base = args.api_base or model_config.get(
        "api_base", os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    )

    # Validate required parameters
    if not api_key:
        print(
            "Error: API key is required. Set it via --api-key-env-var, settings.json, or OPENROUTER_API_KEY/OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename="logs/vibeagent.log",
        filemode="w",
    )
    stdout_logger = logging.getLogger("STDOUT")
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    stderr_logger = logging.getLogger("STDERR")
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    print("--- VibeAgent session started, logging to vibeagent.log ---")

    app = ChatApp(model_id=model_id, api_key=api_key, api_base=api_base)
    app.settings = settings  # Set the settings after instantiation
    app.run()
