import json
from typing import Any

import litellm
from pydantic import BaseModel, Field

litellm.drop_params = True


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

    def __init__(
        self,
        model: str = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        state: dict | None = None,
        max_steps: int = 30,
        **completion_args: dict[str, Any],
    ):
        """Init agent."""
        self.completion_args = {"model": model, **completion_args}
        self.tools = tools or []
        self.tools = {tool.__name__: tool for tool in self.tools}
        self.tool_schemas = [tool.get_schema() for tool in self.tools.values()]
        self.messages = messages or []
        if system_prompt is not None:
            self.messages = [
                {"role": "system", "content": system_prompt},
                *self.messages,
            ]
        self.state = state or {}
        self.r = None
        self.max_steps = max_steps

    @property
    def sanitized_messages(self):
        """Strip invalid fields from messages."""
        sanitized_messages = []
        for message in self.messages:
            sanitized_message = {
                "role": message["role"],
                "content": message["content"],
            }
            if "tool_calls" in message:
                sanitized_message["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                sanitized_message["tool_call_id"] = message["tool_call_id"]
            sanitized_messages.append(sanitized_message)
        return sanitized_messages

    @property
    def tool_history(self):
        """Get the tool history."""
        return [
            message for message in self.messages if "tool_call_id" in message
        ]

    def step(
        self,
        messages: list[dict[str, str]] | str | None = None,
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
        self.r = litellm.completion(**completion_args)
        response_message = dict(self.r.choices[0].message)
        if response_message.get("tool_calls"):
            response_message["tool_calls"] = [
                dict(tool_call, function=dict(tool_call.function))
                for tool_call in response_message["tool_calls"]
            ]
        self.messages.append(response_message)

        # Run tools if present
        if response_message.get("tool_calls"):
            for tool_call in response_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool = self.tools[tool_name]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_response = tool(**tool_args)(self) or "Success"
                self.messages.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": tool_response,
                        "name": tool_name,
                        "args": tool_args,
                    }
                )
        return response_message

    def run(
        self,
        messages: list[dict[str, str]] | str | None = None,
        **completion_args: dict,
    ):
        """Run a sequence of steps including tool calls."""
        for _ in range(self.max_steps):
            response_message = self.step(messages, **completion_args)
            messages = None  # Only pass messages on the first step
            if self.r.choices[0].finish_reason != "tool_calls":
                break
        return response_message


__all__ = ["Agent", "Tool", "Field"]
