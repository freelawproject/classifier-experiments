# LLM Tools

To use LLM tools you should configure the appropriate API keys in your environment. Any providers / models supported by [litellm](https://docs.litellm.ai/docs/providers) are supported.

**Amazon Bedrock**
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
```

**OpenAI**
```
OPENAI_API_KEY=...
```

**Anthropic**
```
ANTHROPIC_API_KEY=...
```

## Agents

Use the `Agent` class to use LLMs with tools. For a simple chat use `agent.step`:

```python
from clx.llm import Agent

agent = Agent()
response = agent.step("Hello!")
print(response)

"""
Output:
{
  'role': 'assistant',
  'content': 'Hello, how can I help you today?',
  ...
}
"""
```

### Tool Calling

Extend the `Tool` class, add fields with `Field`, and implement the `__call__` method to define your tool. For example:

```python
from clx.llm import Field, Tool

class ExtractVerbTool(Tool):
    """Use this to extract verbs from a sentence."""
    verbs: list[str] = Field(description="List of extracted verbs.")

    def __call__(self, agent: Agent) -> str:
        """Tool call."""
        # You can store arbitrary data from your tool call in agent.state
        agent.state["verbs"] = self.verbs
```

Then you can use your tool with an agent. Here, the `tool_choice="required"` forces the agent to use its tool and prevents it from responding directly:

```python
agent = Agent(tools=[ExtractVerbTool], tool_choice="required")
agent.step("Extract the verbs: 'The cat chased the dog.'")
print(agent.state)

"""
Output:
{'verbs': ['chased']}
"""
```

For multi-step agents, you can use the `agent.run` method to execute a sequence of steps including tool calls. The agent will receive the results of the tool call and decide whether to continue or stop. This is useful for agents that may need to query external resources or get some feedback from the tool call.

For example:

```python
class WeatherTool(Tool):
    """Check the weather."""
    location: str = Field(description="Location to check.")

    def __call__(self, agent: Agent) -> str:
        """Tool call."""
        # Text returned by your tool call will be sent to the agent.
        temp = 72 # i.e. get_temp_from_weather_api(self.location)
        return f"The weather in {self.location} is {temp}°F."

agent = Agent(tools=[WeatherTool])
response = agent.run("What's the weather in North Carolina?")
print(response)

"""
Output:
{
  'role': 'assistant',
  'content': 'It is 72 degrees Fahrenheit in North Carolina today.',
  ...
}
"""
```

### Agent Attributes

The `Agent` class has some attributes that are useful to know about:

- `messages`: The conversation history, include tool calls and responses.
- `state`: The state of the agent, up to you to populate and use.
- `tools`: The tools available to the agent.
- `r`: The raw response from the last completion call.

Arbitrary completion arguments can be passed to the `Agent` initialization (e.g. `model`, `max_tokens`, `temperature`, `tool_choice`, etc.). You can also pass them to the `agent.step` and `agent.run` to override the defaults you set in the initialization.
