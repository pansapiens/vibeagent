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
import shlex
from datetime import datetime
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


try:
    from openai.types.chat.chat_completion import ChatCompletion
except ImportError:
    ChatCompletion = None  # type: ignore


# Custom JSON encoder to handle non-serializable objects from smolagents
class VibeAgentJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ChatMessage):
            return o.dict()
        if ChatCompletion and isinstance(o, ChatCompletion):
            return o.model_dump()
        return super().default(o)


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

    Input.shell-mode {
        border: tall $warning;
        background: $panel;
        color: $foreground;
    }

    Input.shell-mode:focus {
        border: tall $warning;
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

    class ShellResponse(Message):
        """A message containing the shell command response."""

        def __init__(self, response: str) -> None:
            super().__init__()
            self.response = response

    def __init__(
        self,
        model_config: dict,
        initial_model_id: str,
        config_dir: Path,
        log_dir: Path,
        data_dir: Path,
    ):
        super().__init__()
        self.title = "vibeagent"
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
        self.default_compress_strategy = self.settings.get(
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
        self.data_dir = data_dir
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Auto-save functionality (will be initialized in on_mount)
        self.auto_save_enabled = False
        self.auto_save_session_name = None

        # Shell session management
        # Initialize shell session
        self.shell_session = ShellSession(self.settings)

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

    def _get_session_path(self, name: str | None) -> Path:
        """Resolves a session name or path into a full Path object."""
        if name and name.endswith(".json"):
            # Treat as a relative or absolute path
            return Path(name)

        session_name = name if name is not None else "_default"
        return self.sessions_dir / f"{session_name}.json"

    def _display_ui_message(
        self,
        message: str,
        is_error: bool = False,
        from_thread: bool = False,
        render_as_markdown: bool = False,
        exc_info: bool = False,
    ):
        """Helper to display info or error messages in the UI and log errors."""
        if is_error:
            logging.error(message, exc_info=exc_info)
            message_class = "error-message"
        else:
            message_class = "info-message"

        widget = (
            Markdown(message, classes=message_class)
            if render_as_markdown
            else Static(message, classes=message_class)
        )

        if from_thread:
            self.call_from_thread(self.query_one("#chat-history").mount, widget)
            self.call_from_thread(self.query_one("#chat-history").scroll_end)
        else:
            self.query_one("#chat-history").mount(widget)
            self.query_one("#chat-history").scroll_end()

    def _validate_session_file(
        self, path: Path, operation: str, from_thread: bool = False
    ) -> bool:
        """Validates session file existence for operations that need it."""
        if not path.exists():
            self._display_ui_message(
                f"Session file not found: {path}",
                is_error=True,
                from_thread=from_thread,
            )
            return False
        return True

    def _capture_ui_history(self) -> list[dict]:
        """Captures the current UI chat history."""
        ui_history = []
        chat_history_widget = self.query_one("#chat-history")

        for child in chat_history_widget.children:
            if "user-message" in child.classes and isinstance(child, Static):
                # Strip the leading "> "
                content = str(child.renderable).split("> ", 1)[-1]
                ui_history.append({"role": "user", "content": content})
            elif "agent-message" in child.classes and isinstance(child, Markdown):
                ui_history.append({"role": "agent", "content": child._markdown})

        return ui_history

    def _capture_agent_memory(self) -> list[dict]:
        """Captures the current agent memory state."""
        if not self.agent:
            return []
        return [step.dict() for step in self.agent.memory.steps]

    def _reconstruct_agent_memory(self, agent_memory_steps: list[dict]) -> None:
        """Reconstructs agent memory from saved session data."""
        if not self.agent:
            return

        from smolagents.memory import ActionStep, PlanningStep, TaskStep, ToolCall
        from smolagents.models import ChatMessage, TokenUsage
        from smolagents.monitoring import Timing

        loaded_steps = []
        for step_dict in agent_memory_steps:
            # Reconstruct complex nested objects from their dicts
            if "timing" in step_dict and step_dict["timing"]:
                timing_data = step_dict["timing"]
                if "duration" in timing_data:
                    del timing_data[
                        "duration"
                    ]  # Don't pass calculated property to constructor
                step_dict["timing"] = Timing(**timing_data)

            if "token_usage" in step_dict and step_dict["token_usage"]:
                token_usage_data = step_dict["token_usage"]
                if "total_tokens" in token_usage_data:
                    del token_usage_data["total_tokens"]
                step_dict["token_usage"] = TokenUsage(**token_usage_data)

            if "tool_calls" in step_dict and step_dict["tool_calls"]:
                reconstructed_calls = []
                for tc_dict in step_dict["tool_calls"]:
                    func_data = tc_dict.get("function", {})
                    reconstructed_calls.append(
                        ToolCall(
                            id=tc_dict.get("id"),
                            name=func_data.get("name"),
                            arguments=func_data.get("arguments"),
                        )
                    )
                step_dict["tool_calls"] = reconstructed_calls

            if (
                "model_input_messages" in step_dict
                and step_dict["model_input_messages"]
            ):
                step_dict["model_input_messages"] = [
                    ChatMessage.from_dict(m) for m in step_dict["model_input_messages"]
                ]

            if (
                "model_output_message" in step_dict
                and step_dict["model_output_message"]
            ):
                step_dict["model_output_message"] = ChatMessage.from_dict(
                    step_dict["model_output_message"]
                )

            # Remove fields that can't be easily serialized/deserialized
            step_dict.pop("error", None)
            step_dict.pop("observations_images", None)
            step_dict.pop("task_images", None)

            # Differentiate step type based on unique keys
            if "step_number" in step_dict:
                loaded_steps.append(ActionStep(**step_dict))
            elif "plan" in step_dict:
                loaded_steps.append(PlanningStep(**step_dict))
            elif "task" in step_dict:
                loaded_steps.append(TaskStep(**step_dict))

        self.agent.memory.steps = loaded_steps
        action_step_count = sum(
            1 for step in loaded_steps if isinstance(step, ActionStep)
        )
        self.agent.step_number = action_step_count + 1

    def _restore_ui_history(self, ui_history: list[dict]) -> None:
        """Restores UI chat history from session data."""
        chat_history_widget = self.query_one("#chat-history")
        self.call_from_thread(chat_history_widget.remove_children)

        for message in ui_history:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                self.call_from_thread(
                    chat_history_widget.mount,
                    Static(f"> {content}", classes="user-message"),
                )
            elif role == "agent":
                self.call_from_thread(
                    chat_history_widget.mount,
                    Markdown(content, classes="agent-message"),
                )

    def _create_session_data(self) -> dict:
        """Creates session data dictionary from current state."""
        return {
            "metadata": {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "model_id": self.model_id,
            },
            "ui_history": self._capture_ui_history(),
            "agent_memory_steps": self._capture_agent_memory(),
            "command_history": self.command_history,
        }

    def save_session(self, name: str | None):
        """Saves the current chat session to a file."""
        try:
            path = self._get_session_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)

            session_data = self._create_session_data()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, cls=VibeAgentJSONEncoder)

            self._display_ui_message(f"Session saved to {path}")

        except Exception as e:
            self._display_ui_message(
                f"Error saving session: {e}", is_error=True, exc_info=True
            )

    def _auto_save_session(self):
        """Automatically saves the current session if auto-save is enabled."""
        if not self.auto_save_enabled:
            return

        try:
            path = self._get_session_path(self.auto_save_session_name)
            path.parent.mkdir(parents=True, exist_ok=True)

            session_data = self._create_session_data()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, cls=VibeAgentJSONEncoder)

        except Exception as e:
            # Don't display error messages for auto-save to avoid cluttering the UI
            logging.error(f"Auto-save error: {e}", exc_info=True)

    def delete_session(self, name: str | None):
        """Deletes a saved session file."""
        if name is None:
            self._display_ui_message(
                "Usage: /delete <session_name>", is_error=True, from_thread=True
            )
            return

        try:
            path = self._get_session_path(name)
            if not self._validate_session_file(path, "delete", from_thread=True):
                return

            path.unlink()
            self._display_ui_message(f"Session '{name}' deleted.", from_thread=True)

        except Exception as e:
            self._display_ui_message(
                f"Error deleting session: {e}",
                is_error=True,
                from_thread=True,
                exc_info=True,
            )

    def load_session(self, name: str | None):
        """Loads a chat session from a file."""
        try:
            path = self._get_session_path(name)
            if not self._validate_session_file(path, "load", from_thread=True):
                return

            with open(path, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Restore agent memory
            self._reconstruct_agent_memory(session_data.get("agent_memory_steps", []))

            # Restore UI
            self._restore_ui_history(session_data.get("ui_history", []))

            # Restore command history
            self._restore_command_history(session_data.get("command_history", []))

            # Restore model if different
            model_id = session_data.get("metadata", {}).get("model_id")
            if model_id and model_id != self.model_id:
                self.call_from_thread(self.update_agent_with_new_model, model_id)
                self._display_ui_message(
                    f"Switched to session model: {model_id}", from_thread=True
                )

            self._display_ui_message(f"Session '{path.stem}' loaded.", from_thread=True)

        except Exception as e:
            self._display_ui_message(
                f"Error loading session: {e}",
                is_error=True,
                from_thread=True,
                exc_info=True,
            )

    def _restore_command_history(self, command_history: list[str]) -> None:
        """Restores the command history from session data."""
        self.command_history = command_history
        self.history_index = len(self.command_history)

    async def fetch_models(self) -> None:
        """Fetches available models from all enabled OpenAI-compatible endpoints."""
        from openai import AsyncOpenAI

        endpoints = self.model_config.get("endpoints", {})
        if not endpoints:
            self._display_ui_message(
                "No model endpoints configured in settings.", is_error=True
            )
            return

        all_models_by_provider = {}
        model_id_counts = {}

        for provider_name, config in endpoints.items():
            if not config.get("enabled", True):
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
                self._display_ui_message(
                    f"Error fetching models from {provider_name}: {e}",
                    is_error=True,
                    exc_info=True,
                )

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

        self._display_ui_message(
            f"Found {len(self.available_models)} available models from {len(all_models_by_provider)} provider(s)."
        )

    def on_mount(self) -> None:
        """Called when the app is mounted. Sets up the agent and MCP connections."""
        # Register and set the custom theme
        self.register_theme(cyberpunk_theme)
        self.theme = "cyberpunk"

        # Initialize title and input styling
        self._update_title_for_shell_mode(False)
        self._update_input_style_for_shell_mode(False)

        # Initialize shell session
        self.shell_session = ShellSession(self.settings)

        # Initialize auto-save functionality
        self.auto_save_enabled = self.settings.get("autoSave", True)
        if self.auto_save_enabled:
            # Generate auto-save session name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.auto_save_session_name = f"_autosave_{timestamp}"

        # provider = trace.get_tracer_provider()
        # provider.add_span_processor(SimpleSpanProcessor(self.TextualSpanExporter(self)))
        self._display_ui_message("Initializing agent and connecting to tool servers...")
        self.favorite_models = self.settings.get("favoriteModels", [])
        self.run_worker(self.fetch_models, thread=True)

        # Check and pull container image in main thread before starting worker
        container_settings = self.settings.get("containers", {})
        if container_settings.get("enabled", False):
            if not self._check_and_pull_container_image():
                self._display_ui_message(
                    "Failed to pull container image. MCP servers may not work properly. Check that Docker/Apptainer is installed and running.",
                    is_error=True,
                )
                # Continue anyway - the worker will handle MCP server setup

        self.is_loading = True
        self.query_one(Input).placeholder = "Initializing agent..."
        self.run_worker(
            partial(self.setup_agent_and_tools, self.settings),
            name="setup_agent",
            thread=True,
        )
        chat_history = self.query_one("#chat-history")
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

    def _wrap_command_for_docker(
        self, server_name: str, command: str, args: list[str], container_settings: dict
    ) -> tuple[str, list[str]]:
        """Wraps a command to be run inside a Docker container."""
        image = container_settings.get("image")
        home_mount_point = container_settings.get("home_mount_point")

        docker_cmd = "docker"
        docker_args = [
            "run",
            "--rm",
            "-i",
        ]  # Run, remove on exit, and interactive for stdin

        # Match host user UID/GID to avoid permission issues
        if sys.platform in ["linux", "darwin"]:
            uid = os.getuid()
            gid = os.getgid()
            docker_args.extend(["--user", f"{uid}:{gid}"])

        # Create and mount an isolated home directory for the server's dependencies
        mcp_servers_data_dir = self.data_dir / "mcp-servers"
        server_home_dir = mcp_servers_data_dir / server_name
        server_home_dir.mkdir(parents=True, exist_ok=True)
        docker_args.extend(["-v", f"{server_home_dir.resolve()}:{home_mount_point}"])

        # Mount allowedPaths read-write
        allowed_paths = self.settings.get("allowedPaths", [])
        resolved_workdir = None
        for path_str in allowed_paths:
            path = Path(path_str).resolve()
            if path.exists():
                docker_args.extend(["-v", f"{path}:{path}:rw"])
                if (
                    resolved_workdir is None
                ):  # Set workdir to the first valid allowed path
                    resolved_workdir = path

        # Set workdir. If no allowed paths, use home directory inside container
        if resolved_workdir:
            docker_args.extend(["--workdir", str(resolved_workdir)])
        else:
            docker_args.extend(["--workdir", home_mount_point])

        # The container image to use
        docker_args.append(image)

        # The original command and its arguments
        docker_args.append(command)
        docker_args.extend(args)

        logging.info(f"Wrapped command for '{server_name}' to run in Docker.")
        return docker_cmd, docker_args

    def _wrap_command_for_apptainer(
        self, server_name: str, command: str, args: list[str], container_settings: dict
    ) -> tuple[str, list[str]]:
        """Wraps a command to be run inside an Apptainer container."""
        image = container_settings.get("image")
        home_mount_point = container_settings.get("home_mount_point")

        apptainer_cmd = "apptainer"
        apptainer_args = ["run", "--cleanenv"]  # --cleanenv for better isolation

        # Create and mount an isolated home directory for the server's dependencies
        mcp_servers_data_dir = self.data_dir / "mcp-servers"
        server_home_dir = mcp_servers_data_dir / server_name
        server_home_dir.mkdir(parents=True, exist_ok=True)
        apptainer_args.extend(
            ["--home", f"{server_home_dir.resolve()}:{home_mount_point}"]
        )

        # Mount allowedPaths read-write
        allowed_paths = self.settings.get("allowedPaths", [])
        resolved_workdir = None
        for path_str in allowed_paths:
            path = Path(path_str).resolve()
            if path.exists():
                apptainer_args.extend(["--bind", f"{path}:{path}:rw"])
                if resolved_workdir is None:  # Set workdir to first valid path
                    resolved_workdir = path

        # Set workdir
        if resolved_workdir:
            apptainer_args.extend(["--pwd", str(resolved_workdir)])

        # Image URI must be specified for Apptainer
        apptainer_args.append(f"docker://{image}")

        # Original command
        apptainer_args.append(command)
        apptainer_args.extend(args)

        logging.info(f"Wrapped command for '{server_name}' to run in Apptainer.")
        return apptainer_cmd, apptainer_args

    def _wrap_command_for_container(
        self, server_name: str, server_config: dict
    ) -> tuple[str, list[str]]:
        """Wraps a command to be run inside a container if enabled."""
        container_settings = self.settings.get("containers", {})

        command = str(server_config.get("command"))
        args = server_config.get("args", [])

        # Only wrap if containers are enabled
        if not container_settings.get("enabled", False):
            return command, args

        # Check if command is already a container command to avoid double-wrapping
        if command in ["docker", "apptainer"]:
            return command, args

        full_command_str = f"{command} {' '.join(args)}"
        if "docker run" in full_command_str or "apptainer run" in full_command_str:
            return command, args

        engine = container_settings.get("engine", "docker")
        image = container_settings.get("image")
        home_mount_point = container_settings.get("home_mount_point")

        if not image or not home_mount_point:
            logging.warning(
                f"Container for '{server_name}' is not configured properly (missing image or home_mount_point). Running on host."
            )
            return command, args

        common_args = {
            "server_name": server_name,
            "command": command,
            "args": args,
            "container_settings": container_settings,
        }

        if engine == "docker":
            return self._wrap_command_for_docker(**common_args)
        elif engine == "apptainer":
            return self._wrap_command_for_apptainer(**common_args)
        else:
            logging.warning(
                f"Container engine '{engine}' not supported. Running '{server_name}' on host."
            )
            return command, args

    def _check_container_engine_available(self, engine: str) -> bool:
        """Check if the specified container engine is available."""
        import subprocess

        try:
            if engine == "docker":
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
            elif engine == "apptainer":
                result = subprocess.run(
                    ["apptainer", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
            else:
                return False
        except Exception:
            return False

    def _check_and_pull_container_image(self) -> bool:
        """Check if container image exists and pull it if needed. Returns True if successful."""
        container_settings = self.settings.get("containers", {})

        if not container_settings.get("enabled", False):
            return True

        engine = container_settings.get("engine", "docker")
        image = container_settings.get("image")

        if not image:
            self._display_ui_message(
                "Container enabled but no image specified in settings",
                is_error=True,
            )
            return False

        # Check if the container engine is available
        if not self._check_container_engine_available(engine):
            self._display_ui_message(
                f"Container engine '{engine}' is not available. Please install {engine} and ensure it's in your PATH.",
                is_error=True,
            )
            return False

        try:
            if engine == "docker":
                return self._check_and_pull_docker_image(image)
            elif engine == "apptainer":
                return self._check_and_pull_apptainer_image(image)
            else:
                self._display_ui_message(
                    f"Unsupported container engine: {engine}",
                    is_error=True,
                )
                return False
        except Exception as e:
            self._display_ui_message(
                f"Error checking/pulling container image: {e}",
                is_error=True,
            )
            return False

    def _check_and_pull_docker_image(self, image: str) -> bool:
        """Check if Docker image exists and pull it if needed."""
        import subprocess

        try:
            # Check if image exists locally
            result = subprocess.run(
                ["docker", "images", "-q", image],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                logging.info(f"Docker image {image} already exists locally")
                return True

            # Image doesn't exist, pull it
            self._display_ui_message(f"Pulling Docker image {image}...")

            # Add loading indicator
            chat_history = self.query_one("#chat-history")
            loading_indicator = LoadingIndicator(classes="agent-thinking")
            chat_history.mount(loading_indicator)
            chat_history.scroll_end()

            try:
                result = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout for pull
                )

                # Remove loading indicator
                loading_indicator.remove()

                if result.returncode == 0:
                    return True
                else:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    self._display_ui_message(
                        f"Failed to pull Docker image {image}: {error_msg}",
                        is_error=True,
                    )
                    return False

            except subprocess.TimeoutExpired:
                loading_indicator.remove()
                self._display_ui_message(
                    f"Timeout pulling Docker image {image}",
                    is_error=True,
                )
                return False

        except Exception as e:
            self._display_ui_message(
                f"Error checking/pulling Docker image {image}: {e}",
                is_error=True,
            )
            return False

    def _check_and_pull_apptainer_image(self, image: str) -> bool:
        """Pull Apptainer image (always pulls since cache checking is unreliable)."""
        import subprocess

        try:
            # Always pull Apptainer images since cache checking is unreliable
            self._display_ui_message(f"Pulling Apptainer image docker://{image}...")

            # Add loading indicator
            chat_history = self.query_one("#chat-history")
            loading_indicator = LoadingIndicator(classes="agent-thinking")
            chat_history.mount(loading_indicator)
            chat_history.scroll_end()

            try:
                # Use apptainer run with a no-op to pull and cache the image without creating a .sif file
                result = subprocess.run(
                    ["apptainer", "run", f"docker://{image}", "echo", "Done!"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout for pull
                )

                # Remove loading indicator
                loading_indicator.remove()

                if result.returncode == 0:
                    return True
                else:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    self._display_ui_message(
                        f"Failed to pull Apptainer image docker://{image}: {error_msg}",
                        is_error=True,
                    )
                    return False

            except subprocess.TimeoutExpired:
                loading_indicator.remove()
                self._display_ui_message(
                    f"Timeout pulling Apptainer image docker://{image}",
                    is_error=True,
                )
                return False

        except Exception as e:
            self._display_ui_message(
                f"Error checking/pulling Apptainer image {image}: {e}",
                is_error=True,
            )
            return False

    def setup_agent_and_tools(self, settings):
        """Initializes the MCP client and the agent with all available tools."""
        logging.info("Starting agent and tool setup...")

        # Check if containers are enabled and if we should proceed with MCP servers
        container_settings = self.settings.get("containers", {})
        containers_enabled = container_settings.get("enabled", False)

        if containers_enabled:
            # If containers are enabled but we're in a worker thread, we can't check/pull images here
            # The image pulling should have been done in the main thread before this worker started
            logging.info("Containers are enabled - MCP servers will run in containers")

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
                # Show message that we're starting MCP servers
                self._display_ui_message("Starting MCP servers...", from_thread=True)
                
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
                    if not config.get("enabled", True):
                        logging.info(f"Skipping disabled MCP server: {name}")
                        continue

                    command, args = self._wrap_command_for_container(name, config)
                    # Allow both "environment" and "env" as equivalent keys
                    env = config.get("environment", config.get("env", {}))
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
                            command=command,
                            args=args,
                            env=full_env,
                        )
                        logging.info(f"[{name}] Creating MCPClient...")
                        client = MCPClient(
                            [server_param],
                        )
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

                self._display_ui_message(
                    f"Error connecting to MCP servers: {e}",
                    is_error=True,
                    from_thread=True,
                    exc_info=True,
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
                self._display_ui_message(
                    f"Error: Details for model '{model_id}' not found.", is_error=True
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

        # Update title with model ID
        self.title = f"vibeagent ({self.model_id})"

        if not startup:
            self._display_ui_message(f"Switched to model: {self.model_id}")
            logging.info(f"Model switched to: {self.model_id}")

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when a worker's state changes."""
        if event.worker.name == "setup_agent":
            # Remove the loading indicator when the setup worker finishes
            if event.worker.state == WorkerState.SUCCESS:
                # The worker is done, update the UI.
                self.is_loading = False
                logging.info(f"Model initialized: {self.model_id}")
                self._display_ui_message(
                    f"Agent is ready. Model: {self.model_id}. Type your message and press Enter."
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
                self._display_ui_message(
                    f"Agent setup failed: {event.worker.error}",
                    is_error=True,
                    exc_info=True,
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

                # Use shell command pwd to get current working directory
                import subprocess

                try:
                    result = subprocess.run(
                        ["pwd"],
                        capture_output=True,
                        text=True,
                        cwd=(
                            self.shell_working_dir
                            if hasattr(self, "shell_working_dir")
                            else None
                        ),
                        timeout=5,
                    )
                    if result.returncode == 0:
                        current_dir = result.stdout.strip()
                    else:
                        current_dir = os.getcwd()  # Fallback to os.getcwd()
                except Exception:
                    current_dir = os.getcwd()  # Fallback to os.getcwd()

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

        # Auto-save after agent response
        self._auto_save_session()

    def on_chat_app_shell_response(self, message: ShellResponse) -> None:
        """Handles shell command responses."""
        # If the worker is None, it means the job was cancelled.
        if self.agent_worker is None:
            return

        self.is_loading = False
        self.agent_worker = None
        self.query_one(Input).placeholder = "Ask the agent to do something..."

        chat_history = self.query_one("#chat-history")
        try:
            chat_history.query(".agent-thinking").last().remove()
        except Exception:
            pass  # It might have already been removed

        response_text = message.response
        # Only mount a response widget if there's actual content
        if response_text.strip():
            chat_history.mount(Markdown(response_text, classes="agent-message"))
            chat_history.scroll_end()

        # Auto-save after shell response
        self._auto_save_session()

    def list_sessions(self) -> list[str]:
        """Lists available session files in the sessions directory, excluding '_default'."""
        if not self.sessions_dir.exists():
            return []
        return [
            f.stem for f in self.sessions_dir.glob("*.json") if f.stem != "_default"
        ]

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
            elif command == "/save":
                return [DropdownItem(f"/save {name}") for name in self.list_sessions()]
            elif command == "/load":
                return [DropdownItem(f"/load {name}") for name in self.list_sessions()]
            elif command == "/delete":
                return [
                    DropdownItem(f"/delete {name}") for name in self.list_sessions()
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
                "/save",
                "/load",
                "/delete",
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
        if command == "/save":
            self.save_session(arg)
            return True
        if command == "/load":
            self.run_worker(partial(self.load_session, arg), thread=True)
            return True
        if command == "/delete":
            self.run_worker(partial(self.delete_session, arg), thread=True)
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

            self._display_ui_message(message)
            return True

        if command == "/tools":
            self.list_tools()
            return True

        if command == "/refresh-models":
            self._display_ui_message("Refreshing model list from API...")
            self.run_worker(self.fetch_models, thread=True)
            return True

        if command == "/model":
            if arg:
                self.update_model(arg.strip())
            else:
                self.action_select_model()
            return True
        if command == "/compress":
            strategy = arg.strip() if arg else self.default_compress_strategy
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

        # Check for redirect commands first (!>, !1>, !2>)
        if self._handle_redirect_command(user_message):
            return

        # Check for shell mode (commands starting with '!')
        if self._check_shell_mode(user_message):
            # Execute shell command
            self.is_loading = True
            self.query_one(Input).placeholder = "Executing shell command..."

            # Run shell command in a worker thread
            self.agent_worker = self.run_worker(
                partial(self._execute_shell_command, user_message),
                thread=True,
                name=f"Shell command: {user_message[:30]}",
            )
            return

        if user_message.startswith("/") and self._handle_command(user_message):
            return

        if not self.agent:
            self._display_ui_message(
                "Agent is not yet initialized. Please wait.", is_error=True
            )
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

        if model_id in self.available_models:
            self.call_from_thread(self.update_agent_with_new_model, model_id)
        else:
            # Try to refresh models to see if it's new
            await self.fetch_models()
            if model_id in self.available_models:
                self.call_from_thread(self.update_agent_with_new_model, model_id)
            else:
                self._display_ui_message(
                    f"Model '{model_id}' not found after refreshing. Please check the name or provider.",
                    is_error=True,
                )

    def list_tools(self) -> None:
        """Displays the list of available tools."""
        if not self.agent or not hasattr(self.agent, "tools"):
            self._display_ui_message(
                "No tools available or agent not initialized.", is_error=True
            )
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

        self._display_ui_message(tool_list_str)

    def action_history_prev(self) -> None:
        """Go to the previous command in history."""
        if self.history_index > 0:
            self.history_index -= 1
            input_widget = self.query_one(Input)
            input_widget.value = self.command_history[self.history_index]
            # Set cursor to the end of the input text
            input_widget.cursor_position = len(input_widget.value)

    def action_history_next(self) -> None:
        """Go to the next command in history."""
        input_widget = self.query_one(Input)
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            input_widget.value = self.command_history[self.history_index]
            # Set cursor to the end of the input text
            input_widget.cursor_position = len(input_widget.value)
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            input_widget.value = ""
            # Set cursor to the beginning when clearing
            input_widget.cursor_position = 0

    def action_scroll_up(self) -> None:
        """Scroll the chat history up."""
        self.query_one("#chat-history").scroll_page_up()

    def action_scroll_down(self) -> None:
        """Scroll the chat history down."""
        self.query_one("#chat-history").scroll_page_down()

    def action_select_model(self) -> None:
        """Show the model selection screen."""
        if not self.available_models:
            self._display_ui_message(
                "Model list not available yet. Please try again shortly.",
                is_error=True,
            )
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

            self._display_ui_message("Request cancelled.")
            self.query_one(Input).placeholder = "Ask the agent to do something..."
            chat_history.scroll_end()

    def get_current_context_tokens(self) -> int:
        """Calculates the total token count from the agent's memory."""
        if not self.agent or not self.agent.memory.steps:
            self._display_ui_message("Agent is not initialized.", is_error=True)
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
            self._display_ui_message("Agent is not initialized.", is_error=True)
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

            self._display_ui_message(markdown_content, render_as_markdown=True)

        except Exception as e:
            self._display_ui_message(
                f"Error dumping context: {e}", is_error=True, exc_info=True
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
                f"**Config Path:**   {self.config_dir}\n\n"
                f"**Data Path:**     {self.data_dir}\n\n"
                f"**Logs Path:**     {self.log_dir}\n\n"
                f"**Sessions Path:** {self.sessions_dir}\n\n"
                f"**settings.json:**\n\n"
                f"```json\n{settings_dump}\n```"
            )

            # If container mode is enabled, show the commands
            container_settings = self.settings.get("containers", {})
            if container_settings.get("enabled", False):
                server_configs = self.settings.get("mcpServers", {})
                command_list = ["\n\n**Container Launch Commands:**\n"]
                for name, config in server_configs.items():
                    if not config.get("enabled", True):
                        continue
                    command, args = self._wrap_command_for_container(name, config)
                    # Don't show unwrapped commands
                    if command in ["docker", "apptainer"]:
                        full_command = " ".join(
                            [command] + [shlex.quote(str(arg)) for arg in args]
                        )
                        command_list.append(f"_{name}_: `{full_command}`")

                if len(command_list) > 1:
                    info_text += "\n\n".join(command_list)

            self._display_ui_message(info_text, render_as_markdown=True)

        except Exception as e:
            self._display_ui_message(
                f"Error showing settings: {e}", is_error=True, exc_info=True
            )
        finally:
            chat_history.scroll_end()

    def compress_context(self, strategy: str):
        if not self.agent:
            self._display_ui_message("Agent is not initialized.", is_error=True)
            return

        if strategy not in ["drop_oldest", "middle_out", "summarize"]:
            self._display_ui_message(f"Unknown strategy: {strategy}", is_error=True)
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
            ALWAYS include the actual content of the final_answer.
            You MUST determine the content that the user was interested in and keep that - this may not always be the last message.
            If search results were returned, keep the actual results.
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

            # Use direct model call instead of agent for summarization
            try:
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
                    self._display_ui_message(
                        "Warning: Summarization returned empty result. Using fallback summary.",
                        is_error=True,
                    )
                    summary = f"Conversation summary: Previous conversation has been compressed. Key information has been preserved."

            except Exception as e:
                self._display_ui_message(
                    f"Error during summarization: {e}. Using fallback summary.",
                    is_error=True,
                )
                summary = f"Conversation summary: Previous conversation has been compressed. Key information has been preserved."

            # Get the token usage from the summarizer agent's last step
            summarizer_token_usage = None
            # For direct model calls, we don't have access to token usage from the model
            # We'll use the estimated token count instead

            # Calculate approximate token count for the summary content
            # Use a simple approximation: 1 token ≈ 4 characters for English text
            summary_tokens = len(summary) // 4 if summary else 0

            # Create a simple summary step to replace the history
            timing = Timing(start_time=time.time())
            timing.end_time = time.time()
            summary_step = ActionStep(
                step_number=1,
                timing=timing,
                model_output=summary,  # Put the summary in model_output instead of observations
                observations=None,  # Clear observations to avoid TOOL_RESPONSE message
                token_usage=summarizer_token_usage
                or TokenUsage(
                    input_tokens=summary_tokens, output_tokens=summary_tokens
                ),  # Use summarizer's token usage or estimate from summary content
            )

            # Replace the agent's memory with just the summary
            self.agent.memory.steps = [summary_step]
            self.agent.step_number = 2

            # Show the summary content for debugging
            self._display_ui_message(
                f"Summary created:\n\n{summary[:500]}{'...' if len(summary) > 500 else ''}",
                render_as_markdown=True,
            )

        # Remove status message
        status_widget.remove()

        # Display final result
        new_tokens = self.get_current_context_tokens()
        self._display_ui_message(
            f"Context compressed from {initial_tokens} to {new_tokens} tokens."
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

    def _get_model_by_name(self, model_name: str) -> "Model":
        """Returns a model from the list of available models."""
        for model in self.available_models:
            if model.lower() == model_name.lower():
                return model
        return None

    def _execute_shell_command(self, command: str) -> None:
        """Executes a shell command in the persistent shell session and posts the response."""
        try:
            # Execute command using shell session
            stdout, stderr, returncode = self.shell_session.execute_command(command)

            # Store command info for redirect functionality
            last_cmd, last_stdout, last_stderr, last_rc = (
                self.shell_session.get_last_command_info()
            )
            self.last_shell_command = last_cmd
            self.last_shell_stdout = last_stdout
            self.last_shell_stderr = last_stderr
            self.last_shell_returncode = last_rc

            # Format response
            response_parts = []

            if stdout:
                response_parts.append(stdout)

            if stderr:
                if response_parts:
                    response_parts.append("")  # Empty line separator
                response_parts.append(stderr)

                # Only show exit code if it's non-zero
            if returncode != 0:
                if response_parts:
                    response_parts.append("")  # Empty line separator
                response_parts.append(f"Exit code: {returncode}")

            # Join all response parts
            response = "\n".join(response_parts) if response_parts else ""

            # Always post a response to ensure the UI is notified of completion
            # Format as markdown code block to preserve newlines
            if response:
                formatted_response = f"```\n{response}\n```"
                self.post_message(self.ShellResponse(formatted_response))
            elif returncode != 0:
                # For non-zero exit codes with no output, show just the exit code
                formatted_response = f"```\nExit code: {returncode}\n```"
                self.post_message(self.ShellResponse(formatted_response))
            else:
                # For successful commands with no output (like cd), post empty response
                self.post_message(self.ShellResponse(""))

        except Exception as e:
            self.post_message(self.ShellResponse(f"Error executing command: {e}"))

    def _update_title_for_shell_mode(self, is_shell_mode: bool) -> None:
        """Updates the app title to show current directory when in shell mode."""
        if is_shell_mode:
            # Use shell working directory from shell session
            current_dir = self.shell_session.get_working_directory()
            self.title = f"vibeagent - {current_dir}"
        else:
            # Restore the title with model ID
            self.title = f"vibeagent ({self.model_id})"

    def _check_shell_mode(self, text: str) -> bool:
        """Checks if the input text indicates shell mode (starts with '!' after stripping whitespace)."""
        return text.lstrip().startswith("!")

    def _update_input_style_for_shell_mode(self, is_shell_mode: bool) -> None:
        """Updates the input widget styling to indicate shell mode."""
        input_widget = self.query_one(Input)
        if is_shell_mode:
            input_widget.add_class("shell-mode")
        else:
            input_widget.remove_class("shell-mode")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handles input changes to detect shell mode and update visual indicators."""
        text = event.value
        is_shell_mode = self._check_shell_mode(text)

        # Update input styling
        self._update_input_style_for_shell_mode(is_shell_mode)

        # Update title for shell mode
        self._update_title_for_shell_mode(is_shell_mode)

    def _handle_redirect_command(self, command: str) -> bool:
        """
        Handles redirect commands (!>, !1>, !2>) that send the last command's output to the LLM.
        Returns True if a redirect command was handled, False otherwise.
        """
        # Check if the command starts with a redirect command
        stripped_command = command.strip()
        redirect_command = None

        if stripped_command.startswith("!>"):
            redirect_command = "!>"
        elif stripped_command.startswith("!1>"):
            redirect_command = "!1>"
        elif stripped_command.startswith("!2>"):
            redirect_command = "!2>"
        else:
            return False

        if not self.last_shell_command:
            self._display_ui_message(
                "No previous shell command found to redirect.", is_error=True
            )
            return True

        # Check if there's a user message after the redirect command
        user_message = None
        if len(stripped_command) > len(redirect_command):
            # There's additional text after the redirect command
            user_message = stripped_command[len(redirect_command) :].strip()

        # Build the message to send to the LLM
        message_parts = []

        # Add user message if provided
        if user_message:
            message_parts.append(f"{user_message}\n")

        # Add shell command and output
        message_parts.append(f"$ {self.last_shell_command}\n")

        if redirect_command == "!>" or redirect_command == "!1>":
            # Include stdout
            if self.last_shell_stdout:
                message_parts.append("STDOUT:\n")
                message_parts.append(f"{self.last_shell_stdout}\n")

        if redirect_command == "!>" or redirect_command == "!2>":
            # Include stderr
            if self.last_shell_stderr:
                message_parts.append("STDERR:\n")
                message_parts.append(f"{self.last_shell_stderr}\n")

        # Always include exit code
        message_parts.append(f"EXITCODE: {str(self.last_shell_returncode)}\n")

        # If no output was captured, indicate this
        if len(message_parts) == (
            3 if user_message else 2
        ):  # Only command and exit code
            message_parts.insert(-2, "(No output captured)")

        # Send the message to the LLM
        redirect_message = "\n".join(message_parts)

        # Send to LLM (don't add to chat history again since it's already added in on_input_submitted)
        self.is_loading = True
        self.query_one(Input).placeholder = "Waiting for response..."

        chat_history = self.query_one("#chat-history")
        chat_history.mount(LoadingIndicator(classes="agent-thinking"))
        chat_history.scroll_end()

        self.agent_worker = self.run_worker(
            partial(self.get_agent_response, redirect_message),
            thread=True,
            name=f"Redirect command: {redirect_command}",
        )

        return True


# Embedded default settings
DEFAULT_SETTINGS = {
    "endpoints": {
        "openrouter": {
            "api_key": "$OPENROUTER_API_KEY",
            "api_base": "https://openrouter.ai/api/v1",
            "enabled": True,
        },
        "local-server": {
            "api_key": "not-needed",
            "api_base": "http://localhost:8080/v1",
            "enabled": False,
        },
        "ollama": {
            "api_key": "ollama",
            "api_base": "http://localhost:11434/v1",
            "enabled": False,
        },
        "pollinations": {
            "api_key": "",
            "api_base": "https://text.pollinations.ai/openai",
            "enabled": False,
        },
    },
    "defaultModel": "mistralai/devstral-small:free",
    "contextLength": 16384,
    "contextManagementStrategy": "summarize",
    "autoSave": True,
    "allowCurrentDirectory": True,
    "favoriteModels": [
        "mistralai/devstral-small:free",
        "mistralai/devstral-small",
        "mistralai/mistral-small-3.2-24b-instruct-2506:free",
        "google/gemma-3n-e4b-it",
    ],
    "allowedPaths": ["$HOME/ai_workspace"],
    "containers": {
        "enabled": False,
        "engine": "docker",
        "image": "ghcr.io/pansapiens/vibeagent-mcp:latest",
        "home_mount_point": "/home/agent",
        "sandboxBangShellCommands": False,
    },
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "$HOME/ai_workspace",
            ],
            "enabled": True,
        },
        "shell": {
            "command": "uvx",
            "args": ["mcp-shell-server"],
            "env": {"ALLOW_COMMANDS": "ls,cat,pwd,grep,wc,touch,find,jq"},
            "enabled": True,
        },
        "text-editor": {
            "command": "uvx",
            "args": ["mcp-text-editor"],
            "enabled": True,
        },
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "enabled": True,
        },
        "playwright": {
            "command": "npx",
            "args": ["@executeautomation/playwright-mcp-server"],
            "autoApprove": [],
            "enabled": False,
        },
        "wcgw": {
            "command": "uv",
            "args": ["tool", "run", "--python", "3.12", "wcgw@latest"],
            "enabled": True,
        },
        "searxng": {
            "command": "npx",
            "args": ["-y", "mcp-searxng"],
            "enabled": True,
        },
        "pollinations": {
            "command": "npx",
            "args": ["@pollinations/model-context-protocol"],
            "enabled": True,
        },
    },
}


class ShellSession:
    """Manages a persistent shell session with state capture and restoration."""

    def __init__(self, settings: dict):
        """Initialize the shell session with settings."""
        self.settings = settings
        self.working_dir = None
        self.env = None
        self.functions = {}  # Store function definitions
        self.aliases = {}  # Store aliases
        self.options = {}  # Store shell options

        # Last command state for redirect functionality
        self.last_command = None
        self.last_stdout = None
        self.last_stderr = None
        self.last_returncode = None

        # Initialize the session
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initializes the shell session with working directory and environment."""
        import os

        # Set initial working directory from allowedPaths if available
        allowed_paths = self.settings.get("allowedPaths", [])
        if allowed_paths:
            initial_path = allowed_paths[0]
            # Expand environment variables and user home directory
            initial_path = os.path.expanduser(os.path.expandvars(initial_path))
            if os.path.exists(initial_path) and os.path.isdir(initial_path):
                self.working_dir = initial_path
                # Change the actual working directory to the first allowed path
                try:
                    os.chdir(initial_path)
                    logging.info(f"Changed working directory to: {initial_path}")
                except Exception as e:
                    logging.warning(
                        f"Failed to change working directory to {initial_path}: {e}"
                    )
            else:
                self.working_dir = os.getcwd()
        else:
            self.working_dir = os.getcwd()

        # Initialize environment with current environment
        self.env = os.environ.copy()

        # Capture initial shell state
        self._capture_state()

        logging.info(
            f"Shell session initialized with working directory: {self.working_dir}"
        )

    def _capture_state(self) -> None:
        """Captures current shell state including functions, aliases, and options."""
        import subprocess

        try:
            # Capture functions
            result = subprocess.run(
                "declare -f",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                env=self.env,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse function definitions
                self.functions = {}
                lines = result.stdout.split("\n")
                current_func = None
                func_lines = []

                for line in lines:
                    if line.startswith("declare -f "):
                        if current_func and func_lines:
                            self.functions[current_func] = "\n".join(func_lines)
                        current_func = line.split(" ")[2]
                        func_lines = [line]
                    elif current_func and line.strip():
                        func_lines.append(line)

                if current_func and func_lines:
                    self.functions[current_func] = "\n".join(func_lines)

            # Capture aliases
            result = subprocess.run(
                "alias",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                env=self.env,
                timeout=10,
            )
            if result.returncode == 0:
                self.aliases = {}
                for line in result.stdout.split("\n"):
                    if line.startswith("alias "):
                        # Parse alias definition
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            alias_name = parts[0].split(" ")[1]
                            alias_value = parts[1].strip("'\"")
                            self.aliases[alias_name] = alias_value

            # Capture shell options
            result = subprocess.run(
                "set",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                env=self.env,
                timeout=10,
            )
            if result.returncode == 0:
                self.options = {}
                for line in result.stdout.split("\n"):
                    if line.startswith("set "):
                        # Parse set options
                        options = line[4:].split()
                        for opt in options:
                            if opt.startswith("-") or opt.startswith("+"):
                                self.options[opt] = True

            logging.info(
                f"Captured shell state: {len(self.functions)} functions, {len(self.aliases)} aliases, {len(self.options)} options"
            )

        except Exception as e:
            logging.warning(f"Failed to capture shell state: {e}")
            # Initialize empty state
            self.functions = {}
            self.aliases = {}
            self.options = {}

    def _create_state_file_path(self) -> Path:
        """Creates a unique state file path in /tmp to prevent race conditions."""
        import tempfile

        # Create a unique filename using process ID and UUID
        unique_id = f"vibeagent_shell_state_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        return Path("/tmp") / unique_id

    def _generate_state_capture_commands(self, state_file_path: Path) -> str:
        """Generates shell commands to capture the complete shell state to a file."""
        # Create a comprehensive state capture script
        state_capture_script = f"""#!/bin/bash
# Capture complete shell state to {state_file_path}

# Create the state file with proper permissions
touch "{state_file_path}"
chmod 600 "{state_file_path}"

# Capture functions
echo "###! FUNCTIONS !###" >> "{state_file_path}"
declare -f >> "{state_file_path}" 2>/dev/null || true

# Capture aliases
echo "###! ALIASES !###" >> "{state_file_path}"
alias >> "{state_file_path}" 2>/dev/null || true

# Capture shell options (but exclude variables that might contain our markers)
echo "###! SHELL_OPTIONS !###" >> "{state_file_path}"
set | grep -v "BASH_EXECUTION_STRING" | grep -v "###!" >> "{state_file_path}" 2>/dev/null || true

# Capture environment variables
echo "###! ENVIRONMENT !###" >> "{state_file_path}"
env >> "{state_file_path}" 2>/dev/null || true

# Capture working directory
echo "###! WORKING_DIRECTORY !###" >> "{state_file_path}"
pwd >> "{state_file_path}" 2>/dev/null || true

# Capture exit code (will be set after command execution)
echo "###! EXIT_CODE !###" >> "{state_file_path}"
echo "$?" >> "{state_file_path}"

# Ensure the file is written
sync "{state_file_path}"
"""
        return state_capture_script

    def _parse_state_file(self, state_file_path: Path) -> dict:
        """Parses the state file and returns a dictionary with all captured state."""
        state = {
            "functions": {},
            "aliases": {},
            "shell_options": {},
            "environment": {},
            "working_directory": None,
            "exit_code": 0,
        }

        try:
            if not state_file_path.exists():
                logging.warning(f"State file {state_file_path} does not exist")
                return state

            with open(state_file_path, "r") as f:
                content = f.read()

                # Parse the state file sections
            sections = content.split("###!")
            current_section = None
            current_content = []

            for section in sections:
                if not section.strip():
                    continue

                lines = section.strip().split("\n")
                if not lines:
                    continue

                # Extract section name from the first line (format: "SECTION !###")
                first_line = lines[0].strip()
                if " !###" in first_line:
                    section_name = first_line.split(" !###")[0].strip()
                    section_content = lines[1:] if len(lines) > 1 else []

                    # Skip if we've already processed this section (avoid duplicates)
                    if section_name in [
                        "FUNCTIONS",
                        "ALIASES",
                        "SHELL_OPTIONS",
                        "ENVIRONMENT",
                        "WORKING_DIRECTORY",
                        "EXIT_CODE",
                    ] and state.get(section_name.lower().replace("_", "")):
                        continue
                else:
                    # Skip sections that don't match the expected format
                    continue

                if section_name == "FUNCTIONS":
                    # Parse function definitions
                    current_func = None
                    func_lines = []

                    for line in section_content:
                        if line.startswith("declare -f "):
                            if current_func and func_lines:
                                state["functions"][current_func] = "\n".join(func_lines)
                            current_func = line.split(" ")[2]
                            func_lines = [line]
                        elif current_func and line.strip():
                            func_lines.append(line)

                    if current_func and func_lines:
                        state["functions"][current_func] = "\n".join(func_lines)

                elif section_name == "ALIASES":
                    # Parse alias definitions
                    for line in section_content:
                        if line.startswith("alias "):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                alias_name = parts[0].split(" ")[1]
                                alias_value = parts[1].strip("'\"")
                                state["aliases"][alias_name] = alias_value

                elif section_name == "SHELL_OPTIONS":
                    # Parse shell options
                    for line in section_content:
                        if line.startswith("set "):
                            options = line[4:].split()
                            for opt in options:
                                if opt.startswith("-") or opt.startswith("+"):
                                    state["shell_options"][opt] = True

                elif section_name == "ENVIRONMENT":
                    # Parse environment variables
                    for line in section_content:
                        if "=" in line:
                            key, value = line.split("=", 1)
                            state["environment"][key] = value

                elif section_name == "WORKING_DIRECTORY":
                    # Parse working directory
                    if section_content:
                        state["working_directory"] = section_content[0].strip()

                elif section_name == "EXIT_CODE":
                    # Parse exit code
                    if section_content:
                        try:
                            state["exit_code"] = int(section_content[0].strip())
                        except ValueError:
                            state["exit_code"] = 0

        except Exception as e:
            logging.warning(f"Failed to parse state file {state_file_path}: {e}")

        return state

    def _update_state_from_file(self, state_file_path: Path) -> None:
        """Updates the shell state by reading from the state file."""
        state = self._parse_state_file(state_file_path)

        # Update shell state with captured values
        if state["functions"]:
            self.functions = state["functions"]

        if state["aliases"]:
            self.aliases = state["aliases"]

        if state["shell_options"]:
            self.options = state["shell_options"]

        if state["environment"]:
            # Update environment variables, preserving existing ones not in the file
            for key, value in state["environment"].items():
                self.env[key] = value
                logging.debug(f"Updated environment variable: {key}={value}")

        if state["working_directory"]:
            # Update working directory if it exists and is accessible
            try:
                if os.path.exists(state["working_directory"]) and os.path.isdir(
                    state["working_directory"]
                ):
                    self.working_dir = state["working_directory"]
                    # Also update PWD in environment
                    self.env["PWD"] = state["working_directory"]
                    logging.info(f"Working directory updated to: {self.working_dir}")
            except Exception as e:
                logging.warning(f"Failed to update working directory: {e}")

        logging.info(
            f"Updated shell state: {len(state['functions'])} functions, {len(state['aliases'])} aliases, {len(state['shell_options'])} options, {len(state['environment'])} env vars, working_dir: {state['working_directory']}"
        )

    def _cleanup_state_file(self, state_file_path: Path) -> None:
        """Cleans up the state file after use."""
        try:
            if state_file_path.exists():
                state_file_path.unlink()
                logging.debug(f"Cleaned up state file: {state_file_path}")
        except Exception as e:
            logging.warning(f"Failed to cleanup state file {state_file_path}: {e}")

    def reset(self) -> None:
        """Reset the shell session state."""
        self.working_dir = None
        self.env = None
        self.functions = {}
        self.aliases = {}
        self.options = {}
        self._initialize_session()

    def execute_command(self, command: str) -> tuple[str, str, int]:
        """Execute a shell command and return (stdout, stderr, returncode)."""
        import subprocess

        try:
            # Remove the leading '!' and strip whitespace
            shell_command = command.lstrip().lstrip("!").strip()

            if not shell_command:
                return "", "Error: No command provided after '!'", 1

            # Expand tilde (~) to home directory in command arguments
            import shlex

            try:
                # Split the command into parts to handle tilde expansion in arguments
                parts = shlex.split(shell_command)
                expanded_parts = []
                for part in parts:
                    # Only expand tilde if it's at the start of a word and not quoted
                    if part.startswith("~") and (
                        len(part) == 1 or part[1] in "/\\" or part[1].isalnum()
                    ):
                        expanded_parts.append(os.path.expanduser(part))
                    else:
                        expanded_parts.append(part)
                shell_command = " ".join(expanded_parts)
            except Exception as e:
                # If parsing fails, fall back to simple expansion
                logging.warning(f"Failed to parse command for tilde expansion: {e}")
                shell_command = os.path.expanduser(shell_command)

            # Store the command for potential redirect
            self.last_command = shell_command

            # Check for exit command
            if shell_command.lower() in ["exit", "quit"]:
                self.reset()
                return (
                    "",
                    "Shell session reset. A new session will be started on the next command.",
                    0,
                )

            # Execute command with current shell state
            try:
                # Create unique state file path
                state_file_path = self._create_state_file_path()

                # Prepare command with state restoration
                state_restore_commands = []

                # Set working directory if available
                if self.working_dir:
                    state_restore_commands.append(f"cd '{self.working_dir}'")

                # Restore functions
                for func_name, func_def in self.functions.items():
                    state_restore_commands.append(func_def)

                # Restore aliases
                for alias_name, alias_value in self.aliases.items():
                    state_restore_commands.append(f"alias {alias_name}='{alias_value}'")

                # Restore shell options
                for option, enabled in self.options.items():
                    if enabled:
                        state_restore_commands.append(f"set {option}")

                # Restore environment variables that differ from parent process
                parent_env = os.environ.copy()
                env_restore_count = 0
                for key, value in self.env.items():
                    if key not in parent_env or parent_env[key] != value:
                        state_restore_commands.append(f"export {key}='{value}'")
                        env_restore_count += 1

                logging.debug(f"Restoring {env_restore_count} environment variables")
                logging.debug(f"Working directory for next command: {self.working_dir}")

                # Generate state capture commands
                state_capture_commands = self._generate_state_capture_commands(
                    state_file_path
                )

                # Combine state restoration, command execution, and state capture
                full_command = "\n".join(
                    [*state_restore_commands, shell_command, state_capture_commands]
                )

                # Check if we need to sandbox the shell
                container_settings = self.settings.get("containers", {})
                sandbox_shell = container_settings.get(
                    "sandboxBangShellCommands", False
                )

                if sandbox_shell:
                    executable, args = self._wrap_command_for_container(
                        full_command, state_file_path
                    )
                    command_to_run = [executable] + args
                else:
                    command_to_run = full_command

                try:
                    result = subprocess.run(
                        command_to_run,
                        # Use shell=True for non-sandboxed (string command), shell=False for sandboxed (list command)
                        shell=not sandbox_shell,
                        capture_output=True,
                        text=True,
                        cwd=self.working_dir if not sandbox_shell else None,
                        env=self.env,
                        timeout=30,
                    )

                    # Store command output for potential redirect
                    self.last_stdout = result.stdout
                    self.last_stderr = result.stderr
                    self.last_returncode = result.returncode

                except FileNotFoundError:
                    if sandbox_shell:
                        return (
                            "",
                            f"Error: Container engine '{container_settings.get('engine', 'docker')}' not found. Please ensure it is installed and in your PATH.",
                            1,
                        )
                    else:
                        raise

                # Update shell state from the captured state file
                try:
                    logging.debug(f"Updating shell state from file: {state_file_path}")
                    self._update_state_from_file(state_file_path)
                    logging.debug(
                        f"Shell state updated - working_dir: {self.working_dir}, env vars: {len(self.env)}"
                    )

                    # Debug: Show some key environment variables
                    if self.env:
                        pwd_val = self.env.get("PWD", "not set")
                        home_val = self.env.get("HOME", "not set")
                        logging.debug(f"PWD: {pwd_val}, HOME: {home_val}")

                except Exception as e:
                    logging.warning(f"Failed to update shell state from file: {e}")

                # Clean up the state file
                self._cleanup_state_file(state_file_path)

                # Format output in terminal-like style
                output_parts = []

                # Combine stdout and stderr in the order they would appear in a terminal
                # Most commands don't produce stderr, so stdout first is usually correct
                if result.stdout:
                    # Remove state capture output from the end of stdout
                    output_lines = result.stdout.split("\n")

                    # Find where the state capture starts (look for the state capture script)
                    state_start_idx = -1
                    for i, line in enumerate(output_lines):
                        if line.strip().startswith(
                            "#!/bin/bash"
                        ) and "Capture complete shell state" in " ".join(
                            output_lines[i : i + 3]
                        ):
                            state_start_idx = i
                            break

                    # Remove state capture output if found
                    if state_start_idx != -1:
                        output_lines = output_lines[:state_start_idx]

                    # Remove trailing empty lines
                    while output_lines and not output_lines[-1].strip():
                        output_lines.pop()

                    if output_lines:
                        output_parts.append("\n".join(output_lines).rstrip())

                if result.stderr:
                    # Add stderr after stdout (typical terminal behavior)
                    if output_parts:
                        output_parts.append("")  # Empty line separator
                    output_parts.append(result.stderr.rstrip())

                # Only show exit code if it's non-zero
                if result.returncode != 0:
                    if output_parts:
                        output_parts.append("")  # Empty line separator
                    output_parts.append(f"Exit code: {result.returncode}")

                # Join all output parts
                response = "\n".join(output_parts) if output_parts else ""

                return response, "", result.returncode

            except subprocess.TimeoutExpired:
                return "", "Command timed out after 30 seconds.", 1

        except Exception as e:
            return "", f"Error executing command: {e}", 1

    def _wrap_command_for_container(
        self, shell_command: str, state_file_path: Path
    ) -> tuple[str, list[str]]:
        """Wraps a shell command to be run inside a container."""
        container_settings = self.settings.get("containers", {})
        engine = container_settings.get("engine", "docker")

        # The command to run inside the container is bash
        command = "bash"
        # The argument to bash is the shell command itself
        args = ["-c", shell_command]

        if engine == "docker":
            return self._wrap_command_for_docker(
                shell_command, container_settings, state_file_path
            )
        elif engine == "apptainer":
            return self._wrap_command_for_apptainer(
                shell_command, container_settings, state_file_path
            )
        else:
            # Should not happen if settings are validated, but as a fallback
            logging.warning(
                f"Container engine '{engine}' not supported for sandboxed shell. Running on host."
            )
            return command, args

    def _wrap_command_for_docker(
        self, shell_command: str, container_settings: dict, state_file_path: Path
    ) -> tuple[str, list[str]]:
        """Wraps a shell command to be run inside a Docker container."""
        image = container_settings.get("image")
        home_mount_point = container_settings.get("home_mount_point")

        docker_cmd = "docker"
        docker_args = [
            "run",
            "--rm",
            "-i",
        ]  # Run, remove on exit, and interactive for stdin

        # Match host user UID/GID to avoid permission issues
        if sys.platform in ["linux", "darwin"]:
            uid = os.getuid()
            gid = os.getgid()
            docker_args.extend(["--user", f"{uid}:{gid}"])

        # Mount allowedPaths read-write
        allowed_paths = self.settings.get("allowedPaths", [])
        resolved_workdir = None
        for path_str in allowed_paths:
            path = Path(path_str).resolve()
            if path.exists():
                docker_args.extend(["-v", f"{path}:{path}:rw"])
                if (
                    resolved_workdir is None
                ):  # Set workdir to the first valid allowed path
                    resolved_workdir = path

        # Mount the state file directory to preserve state across container runs
        # Mount the entire /tmp directory to ensure the state file is accessible
        docker_args.extend(["-v", "/tmp:/tmp:rw"])

        # Set workdir. If no allowed paths, use home directory inside container
        if resolved_workdir:
            docker_args.extend(["--workdir", str(resolved_workdir)])
        else:
            docker_args.extend(["--workdir", home_mount_point])

        # The container image to use
        docker_args.append(image)

        # The shell command to run inside the container
        docker_args.append("bash")
        docker_args.extend(["-c", shell_command])

        logging.info(f"Wrapped shell command to run in Docker: {shell_command}")
        return docker_cmd, docker_args

    def _wrap_command_for_apptainer(
        self, shell_command: str, container_settings: dict, state_file_path: Path
    ) -> tuple[str, list[str]]:
        """Wraps a shell command to be run inside an Apptainer container."""
        image = container_settings.get("image")
        home_mount_point = container_settings.get("home_mount_point")

        apptainer_cmd = "apptainer"
        apptainer_args = [
            "--silent",  # Suppress INFO messages (global flag)
            "run",
            "--cleanenv",  # Clean environment to ensure complete control
        ]

        # Mount allowedPaths read-write
        allowed_paths = self.settings.get("allowedPaths", [])
        resolved_workdir = None
        for path_str in allowed_paths:
            path = Path(path_str).resolve()
            if path.exists():
                apptainer_args.extend(["--bind", f"{path}:{path}:rw"])
                if resolved_workdir is None:  # Set workdir to first valid path
                    resolved_workdir = path

        # Mount the state file directory to preserve state across container runs
        # Mount the entire /tmp directory to ensure the state file is accessible
        apptainer_args.extend(["--bind", "/tmp:/tmp:rw"])

        # Set workdir
        if resolved_workdir:
            apptainer_args.extend(["--pwd", str(resolved_workdir)])

        # Pass environment variables to preserve shell state
        # Use --env flags to explicitly pass environment variables
        for key, value in self.env.items():
            apptainer_args.extend(["--env", f"{key}={value}"])

        # Image URI must be specified for Apptainer
        apptainer_args.append(f"docker://{image}")

        # The shell command to run inside the container
        apptainer_args.append("bash")
        apptainer_args.extend(["-c", shell_command])

        logging.info(f"Wrapped shell command to run in Apptainer: {shell_command}")
        return apptainer_cmd, apptainer_args

    def get_working_directory(self) -> str:
        """Get the current working directory."""
        return self.working_dir if self.working_dir else os.getcwd()

    def get_last_command_info(self) -> tuple[str, str, str, int]:
        """Get information about the last executed command."""
        return (
            self.last_command,
            self.last_stdout,
            self.last_stderr,
            self.last_returncode,
        )


def main():
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
    data_dir = Path(platformdirs.user_data_dir(APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load settings to get model configuration
    def load_settings(config_dir: Path):
        """Load settings from settings.json with environment variable substitution."""
        settings_path = config_dir / "settings.json"
        if not settings_path.exists():
            print(
                f"settings.json not found in {config_dir}, creating from embedded DEFAULT_SETTINGS..."
            )
            try:
                with open(settings_path, "w") as f_settings:
                    json.dump(DEFAULT_SETTINGS, f_settings, indent=2)
            except IOError as e:
                print(
                    f"Error writing settings.json: {e}. Creating empty settings file."
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

                # Add current directory to allowedPaths if allowCurrentDirectory is True
                if config.get("allowCurrentDirectory", True):
                    current_dir = os.getcwd()
                    allowed_paths = config.get("allowedPaths", [])
                    if current_dir not in allowed_paths:
                        # Insert current directory at the beginning of allowedPaths
                        allowed_paths.insert(0, current_dir)
                        config["allowedPaths"] = allowed_paths
                        logging.info(
                            f"Added current directory to allowedPaths: {current_dir}"
                        )

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
        data_dir=data_dir,
    )
    app.settings = settings  # Set the settings after instantiation
    app.run()


if __name__ == "__main__":
    main()
