import simplejson as json

from clx.llm import Agent, Tool


class ExtractTermsTool(Tool):
    """Extract all relevant terms from the text."""

    terms: list[str] = []

    def __call__(self, agent):
        """Extract all terms from the text."""
        agent.state["terms"] = agent.state.get("terms", []) + self.terms


class ExtractVerbAgent(Agent):
    """Agent for extracting verbs from text."""

    default_tools = [ExtractTermsTool]
    default_model = "gemini/gemini-2.5-flash-lite"
    default_messages = [
        {
            "role": "system",
            "content": (
                "The user will enter some text. Use your ExtractTermsTool "
                "to extract all verbs from the text."
            ),
        }
    ]
    default_max_steps = 2
    default_completion_args = {"tool_choice": "required"}


if 0:
    agent = ExtractVerbAgent(model="openai/gpt-5-mini")
    agent.run("The cat chased the dog and the dog chased the cat.")
    print(agent.state)


class ExtractTermsAgent(Agent):
    """Agent for extracting terms given a user task description."""

    default_tools = [ExtractTermsTool]
    default_model = "gemini/gemini-2.5-flash-lite"
    default_max_steps = 2
    default_completion_args = {"tool_choice": "required"}
    on_init_args = ["task_description"]

    def on_init(self, task_description):
        """Prepare the system prompt based on the task description."""
        self.messages = [
            {
                "role": "system",
                "content": (
                    "The user will enter some text. Use your ExtractTermsTool "
                    "to extract all terms from the text based on following task description:"
                    f"{task_description}"
                ),
            }
        ]


if 0:
    agent = ExtractTermsAgent(
        task_description="Extract all the animals from the text."
    )
    agent.run("The cat chased the dog and the dog chased the cat.")
    print(agent.state)


class AppendToWorkingMemory(Tool):
    """Append an item to your working memory."""

    text: str

    def __call__(self, agent):
        """Extract all terms from the text."""
        agent.state["working_memory"] = agent.state.get(
            "working_memory", []
        ) + [self.text]


class WorkingMemoryAgent(Agent):
    """Agent for appending items to working memory."""

    default_tools = [AppendToWorkingMemory]
    default_model = "gemini/gemini-2.5-flash"

    def on_init(self):
        """Init the system prompt."""
        self.on_step(None)

    def on_step(self, response_message):
        """Update system prompt based on working memory."""
        working_memory = "\n".join(
            ["- " + item for item in self.state.get("working_memory", [])]
        )
        system_prompt = (
            "Your job is to interview the user, keeping track of discrete details using your working memory. "
            "Ask the user about themselves. As they respond, use your AppendToWorkingMemory tool to track the details. "
            "You should use your working memory tool at literally every step if you recieve any information from the user. "
            "We will show your working memory below. This will be updated dynamically on each step."
            f"\n\nHere is what you already know about the user:<WORKING_MEMORY>\n{working_memory}\n</WORKING_MEMORY>"
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
            *self.messages[1:],
        ]


if 0:
    agent = WorkingMemoryAgent()
    user_message = "[User has joined the call]"
    while 1:
        response = agent.run(messages=user_message)
        print("\n\nWorking memory:")
        print(json.dumps(agent.state.get("working_memory", []), indent=2))
        print(f"\n\nResponse: {response['content']}")
        user_message = input("User: ")
