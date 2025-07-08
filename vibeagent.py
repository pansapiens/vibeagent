# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "smolagents[litellm,mcp,telemetry,toolkit]",
#   "textual",
#   "python-dotenv",
#   "aider-chat",
# ]
# ///

import logging
import sys
import json
import os
import io
from functools import partial
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, tool, MCPClient
from smolagents.models import OpenAIServerModel
from mcp import StdioServerParameters
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, Markdown
from textual.message import Message

from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

register()
SmolagentsInstrumentor().instrument()


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

    def __init__(self):
        super().__init__()
        self.title = "Smol Agent Chat"
        self.agent = None
        self.mcp_client = None
        self.command_history = []
        self.history_index = -1
        self.settings = {}
        self.mcp_log_files = {}  # Store MCP server log files
        self.model_id = None  # Add this line

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
        self.setup_agent_and_tools()

        # Log the model name during initialization
        logging.info(f"Model initialized: {self.model_id}")

        self.query_one("#chat-history").mount(
            Static(
                f"Agent is ready. Model: {self.model_id}. Type your message and press Enter.",
                classes="info-message",
            )
        )

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

    def setup_agent_and_tools(self):
        """Initializes the MCP client and the agent with all available tools."""
        local_tools = [aider_edit_file]
        mcp_tools = []
        self.settings = self.load_settings()
        server_configs = self.settings.get("mcpServers", {})

        # Create logs directory structure
        logs_dir = os.path.join("logs", "mcp")
        os.makedirs(logs_dir, exist_ok=True)

        server_params = []

        for name, config in server_configs.items():
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

        # 1. Create a dedicated OpenAI Server model object from environment variables
        openai_server_model = OpenAIServerModel(
            model_id=os.getenv("MODEL", "mistralai/devstral-small:free"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        )

        # 2. Store model ID
        self.model_id = openai_server_model.model_id  # Add this line

        # 2. Pass the model object to the agent
        self.agent = ToolCallingAgent(
            model=openai_server_model,
            tools=all_tools,
            add_base_tools=True,
        )

        self.query_one(Header).title = f"Smol Agent Chat ({len(all_tools)} Tools)"

    def load_settings(self):
        """
        Loads server configurations from settings.json.
        If settings.json doesn't exist, it's copied from default-settings.json.
        Environment variables in string values are substituted in memory.
        """
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

    def compose(self) -> ComposeResult:
        """Creates the layout for the chat application."""
        yield Header()
        yield ScrollableContainer(id="chat-history")
        yield Input(placeholder="Ask the agent to do something...")
        yield Footer()

    def get_agent_response(self, user_message: str) -> None:
        """Worker to get response from agent."""
        if self.agent:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename="vibeagent.log",
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

    app = ChatApp()
    app.run()
