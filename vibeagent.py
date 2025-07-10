# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "smolagents[litellm,mcp,telemetry,toolkit]",
#   "textual",
#   "python-dotenv",
#   "aider-chat"
# ]
# ///

import logging
import sys
import json
import os
import io
import argparse
from functools import partial
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, tool, MCPClient
from smolagents.models import OpenAIServerModel
from mcp import StdioServerParameters
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Markdown
from textual.message import Message

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


class ChatApp(App):
    """A textual-based chat interface for a smolagents agent."""

    CSS_PATH = "styles.css"
    BINDINGS = [
        ("up", "history_prev", "Previous command"),
        ("down", "history_next", "Next command"),
    ]

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
        self.mcp_client = None
        self.command_history = []
        self.history_index = -1
        self.settings = {}
        self.mcp_log_files = {}  # Store MCP server log files
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        self.instrumentor = SmolagentsInstrumentor() if TELEMETRY_AVAILABLE else None
        self.telemetry_is_active = False

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

    def on_mount(self) -> None:
        """Called when the app is mounted. Sets up the agent and MCP connections."""
        # provider = trace.get_tracer_provider()
        # provider.add_span_processor(SimpleSpanProcessor(self.TextualSpanExporter(self)))

        self.query_one("#chat-history").mount(
            Static(
                "Initializing agent and connecting to tool servers...",
                classes="info-message",
            )
        )
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
        if self.mcp_client:
            print("Disconnecting from MCP servers...")
            self.mcp_client.disconnect()

        # Close MCP server log files
        for log_file in self.mcp_log_files.values():
            try:
                log_file.close()
            except:
                pass

    def setup_agent_and_tools(self, settings):
        """Initializes the MCP client and the agent with all available tools."""
        local_tools = [aider_edit_file]
        mcp_tools = []
        self.settings = settings
        server_configs = self.settings.get("mcpServers", {})

        # Create logs directory structure
        logs_dir = os.path.join("logs", "mcp")
        os.makedirs(logs_dir, exist_ok=True)

        server_params = []

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

                server_params.append(
                    StdioServerParameters(command=command, args=args, env=full_env)
                )

        if server_params:
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

                self.mcp_client = MCPClient(server_params)
                mcp_tools = self.mcp_client.get_tools()

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

        all_tools = local_tools + mcp_tools

        # Create a dedicated OpenAI Server model object using instance variables
        openai_server_model = OpenAIServerModel(
            model_id=self.model_id,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        # Pass the model object to the agent
        self.agent = ToolCallingAgent(
            model=openai_server_model,
            tools=all_tools,
            add_base_tools=True,
        )

        self.query_one(Header).title = f"Smol Agent Chat ({len(all_tools)} Tools)"

    def compose(self) -> ComposeResult:
        """Creates the layout for the chat application."""
        yield Header()
        yield ScrollableContainer(id="chat-history")
        yield Input(id="input", placeholder="Ask the agent to do something...")
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
        chat_history = self.query_one("#chat-history")
        chat_history.query(".agent-thinking").last().remove()
        chat_history.mount(Markdown(message.response, classes="agent-message"))
        chat_history.scroll_end()

    def on_chat_app_tool_call(self, message: ToolCall) -> None:
        """Handles a tool call message."""
        chat_history = self.query_one("#chat-history")
        chat_history.mount(
            Static(f"Calling tool: {message.tool_name}", classes="tool-call-message")
        )
        chat_history.scroll_end()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles user input and displays the agent's response."""
        user_message = event.value.strip()

        if not user_message:
            return

        chat_history = self.query_one("#chat-history")
        chat_history.mount(Static(f"> {user_message}", classes="user-message"))
        self.query_one(Input).value = ""
        self.command_history.append(user_message)
        self.history_index = len(self.command_history)
        chat_history.scroll_end()

        if user_message.lower() == "/quit":
            self.exit()
            return

        if user_message.lower() == "/tools":
            self.list_tools()
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

        chat_history.mount(Static("Thinking...", classes="agent-thinking"))
        chat_history.scroll_end()

        self.run_worker(
            partial(self.get_agent_response, user_message),
            thread=True,
            name=f"Agent request: {user_message[:30]}",
        )

    def list_tools(self) -> None:
        """Displays the list of available tools."""
        chat_history = self.query_one("#chat-history")
        if self.agent and hasattr(self.agent, "tools"):
            tools_iterable = self.agent.tools
            if isinstance(tools_iterable, dict):
                tools_iterable = tools_iterable.values()

            tool_items = ["[bold]Available tools:[/bold]"]
            for tool in sorted(tools_iterable, key=lambda t: t.name):
                # Guard against tools with no description
                description = getattr(tool, "description", "No description available.")
                first_line = description.strip().split("\n")[0]
                tool_items.append(f"• [bold]{tool.name}[/bold]: {first_line}")
            tool_list_str = "\n".join(tool_items)
            chat_history.mount(Static(tool_list_str, classes="info-message"))
        else:
            chat_history.mount(
                Static(
                    "No tools available or agent not initialized.",
                    classes="error-message",
                )
            )
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

    if not os.path.exists("styles.css"):
        with open("styles.css", "w") as f:
            f.write(
                """
            Screen { background: #1e1e1e; }
            #chat-history { padding: 1; background: #252526; }
            .user-message { background: #2d2d2d; color: #d4d4d4; padding: 1; margin-bottom: 1; border-left: thick #3c3c3c; }
            .agent-message, .agent-thinking, .info-message, .error-message { background: #333333; color: #cccccc; padding: 1; margin-bottom: 1; border-left: thick #4f4f4f; }
            .tool-call-message { color: #f0e68c; padding-left: 1; margin-bottom: 1; }
            .info-message { color: #8c8c8c; }
            .error-message { color: #ff6347; border-left: thick #ff6347;}
            Input { background: #3c3c3c; border: none; color: #cccccc; }
            """
            )

    app = ChatApp(model_id=model_id, api_key=api_key, api_base=api_base)
    app.settings = settings  # Set the settings after instantiation
    app.run()
