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

## DSPy Predictor

DSPy predictors can be used to learn better prompts for predicting structured outputs. The `DSPyPredictor` base class implements a simple text-to-bool signature with a `MIPROv2` optimizer.

To make an un-optimized prediction, simply initialize a `DSPyPredictor` with some simple instructions:

```python
from clx.llm import DSPyPredictor

predictor = DSPyPredictor(instructions="Predict whether the following text is a complaint.")
preds = predictor.predict(texts, num_workers=4)
```

If you have labeled examples, you can optimize your prompt with the `fit` method:

```python
from clx.llm import DSPyPredictor

examples = [
    {"text": "Complaint files by ...", "value": True},
    ...
]

predictor.fit(examples, num_workers=4)
preds = predictor.predict(texts, num_workers=4)
```

You can save and load your predictor with `predictor.save` and `DSPyPredictor.from_config`:

```python
predictor.save("predictor.json")
predictor = DSPyPredictor.from_config("predictor.json")
```

### GEPAPredictor

The `GEPAPredictor` is a lot more powerful. Use this when you have labeled examples with "reasons" that explain the annotation decision.

```
examples = [
    {"text": "Complaint files by ...", "value": True, "reason": "The text is a complaint."},
    {"text": "Answer to complaint ...", "value": False, "reason": "This merely mentions a complaint, but is not a complaint itself."},
    ...
]

predictor = GEPAPredictor(instructions="Predict whether the following text is a complaint.")
predictor.fit(examples, num_workers=4)
preds = predictor.predict(texts, num_workers=4)
```

GEPA uses a student and teacher model to iteratively improve the prompt. You can customize the student model by passing a string or dictionary as the `model` argument. You can also pass `optimizer_args` to customize the optimizer, the `reflection_lm` key being the configuration for the teacher model.

```python
from clx.llm import GEPAPredictor

predictor = GEPAPredictor(
    model={"model": "bedrock/qwen.qwen3-235b-a22b-2507-v1:0", "temperature": 1.0, "max_tokens": 32000},
    optimizer_args={
        "auto": "heavy",
        "reflection_lm": {
            "model": "gemini/gemini-2.5-pro",
            "temperature": 1.0,
            "max_tokens": 32000,
        },
    },
)
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
