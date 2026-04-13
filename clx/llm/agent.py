import logging
import time
from typing import Any, ClassVar

import litellm
import simplejson as json
import tiktoken
from pydantic import BaseModel, Field

litellm.drop_params = True

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_encoding.encode(text))


logger = logging.getLogger("clx.llm")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)


class Tool(BaseModel):
    """Extend pydantic BaseModel with convenience methods for LLM tools."""

    def __call__(self, agent: "Agent") -> str:
        """Implement the tool call here for multi-step agents.

        Execute your tool here. You can store arbitrary data
        in agent.state and you can return a message for the agent.
        """
        pass

    @classmethod
    def get_schema(cls):
        """Export the tool schema."""
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": cls.__doc__,
                "parameters": cls.model_json_schema(),
            },
        }

    class Config:
        """Configuration to allow extra methods on the BaseModel."""

        extra = "allow"


class Agent:
    """A litellm wrapper for tool calling agents."""

    default_model = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
    default_tools: ClassVar[list[Tool]] = []
    default_max_steps = 30
    default_messages: ClassVar[list[dict[str, str]]] = []
    default_completion_args: ClassVar[dict[str, Any]] = {}
    on_init_args: ClassVar[list[str]] = []

    def __init__(
        self,
        model: str = None,
        tools: list[Tool] | None = None,
        messages: list[dict[str, str]] | None = None,
        state: dict | None = None,
        max_steps: int = None,
        **completion_args: dict[str, Any],
    ):
        """Init agent."""
        init_args = {
            arg: completion_args.pop(arg, None) for arg in self.on_init_args
        }
        model = model or self.default_model
        self.completion_args = {
            "model": model,
            **self.default_completion_args,
            **completion_args,
        }
        self.tools = tools or self.default_tools
        self.tools = {tool.__name__: tool for tool in self.tools}
        self.tool_schemas = [tool.get_schema() for tool in self.tools.values()]
        self.messages = messages or self.default_messages
        self.state = state or {}
        self.r = None
        self.max_steps = max_steps or self.default_max_steps
        self.on_init(**init_args)

    def on_init(self, **kwargs):
        """Hook to run after initialization."""
        pass

    # Fields to exclude from messages sent to the LLM.
    _internal_fields = {"name", "args"}

    @property
    def sanitized_messages(self):
        """Strip internal fields from messages, preserving provider fields."""
        return [
            {
                k: v
                for k, v in message.items()
                if k not in self._internal_fields
            }
            for message in self.messages
        ]

    @property
    def tool_history(self):
        """Get the tool history."""
        return [
            message for message in self.messages if "tool_call_id" in message
        ]

    def step(
        self,
        messages: list[dict[str, str]] | str | None = None,
        call_tools: bool = False,
        **completion_args: dict,
    ) -> tuple[dict, str]:
        """Take a single conversation step."""
        # Prepare completion arguments, allow completion_args override
        completion_args = {
            **self.completion_args,
            "tools": self.tool_schemas,
            **completion_args,
        }

        # Convert messages arg to chat template and append to history
        if messages is not None:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            self.messages.extend(messages)
        completion_args["messages"] = self.sanitized_messages

        # Make the completion call
        logger.info("Calling LLM...")
        t0 = time.time()
        self.r = litellm.completion(**completion_args)
        llm_elapsed = time.time() - t0
        response_message = dict(self.r.choices[0].message)

        # Log the call
        model = completion_args.get("model", "?")
        usage = self.r.usage
        tokens = (
            f"{usage.prompt_tokens}→{usage.completion_tokens}"
            if usage
            else "?"
        )
        try:
            cost = (
                f"${litellm.completion_cost(completion_response=self.r):.4f}"
            )
        except Exception:
            cost = "$?"
        tool_names = ""
        if response_message.get("tool_calls"):
            response_message["tool_calls"] = [
                dict(tool_call, function=dict(tool_call.function))
                for tool_call in response_message["tool_calls"]
            ]
            tool_names = " → " + ", ".join(
                tc["function"]["name"] for tc in response_message["tool_calls"]
            )
        logger.info(
            f"{model} | {tokens} tokens | {cost} | "
            f"{llm_elapsed:.1f}s{tool_names}"
        )
        self.messages.append(response_message)

        # Run tools if present
        if response_message.get("tool_calls") and call_tools:
            for tool_call in response_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool = self.tools[tool_name]
                tool_args = json.loads(tool_call["function"]["arguments"])
                t0 = time.time()
                tool_response = tool(**tool_args)(self) or "Success"
                tool_elapsed = time.time() - t0
                logger.info(
                    f"  {tool_name} → {tool_elapsed:.1f}s | "
                    f"{tool_response[:120]}"
                )
                self.messages.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": tool_response,
                        "name": tool_name,
                        "args": tool_args,
                    }
                )

        logger.info("Saving messages to DB...")
        t0 = time.time()
        self.on_step(response_message)
        save_elapsed = time.time() - t0
        if save_elapsed > 0.5:
            logger.info(f"  DB save took {save_elapsed:.1f}s")
        return response_message

    def on_step(self, response_message: dict):
        """Hook to run after a step."""
        pass

    def run(
        self,
        messages: list[dict[str, str]] | str | None = None,
        **completion_args: dict,
    ):
        """Run a sequence of steps including tool calls."""
        for _ in range(self.max_steps):
            response_message = self.step(
                messages, **completion_args, call_tools=True
            )
            messages = None  # Only pass messages on the first step
            if not response_message.get("tool_calls"):
                break
        return response_message

    def __call__(self, **kwargs):
        """Implement custom workflow here."""
        pass


def message_tokens(msg: dict) -> int:
    """Count the token length of a message's content."""
    tokens = 0
    content = msg.get("content")
    if content:
        tokens += count_tokens(content)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        tokens += count_tokens(fn.get("name", ""))
        tokens += count_tokens(fn.get("arguments", ""))
    return tokens


__all__ = ["Agent", "Tool", "Field", "count_tokens", "message_tokens"]
