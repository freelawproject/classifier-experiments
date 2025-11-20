from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel as PydanticBaseModel
from tqdm import tqdm

from clx.llm import Agent, Tool


class SpanAnnotation(PydanticBaseModel):
    """Annotation for a single span."""

    extracted_text: str
    context_string: str


class SpanAnnotationTool(Tool):
    """Extract spans from text."""

    spans: list[SpanAnnotation]

    def __call__(self, agent):
        """Tool call."""
        text = agent.state["text"]
        spans = []
        for span in self.spans:
            if span.context_string not in text:
                agent.state["status"] = "error"
                return f"Error: Context string {span.context_string} not an exact substring of text."
            elif text.count(span.context_string) != 1:
                agent.state["status"] = "error"
                return f"Error: Context string {span.context_string} is not unique in text."
            elif span.extracted_text not in span.context_string:
                agent.state["status"] = "error"
                return f"Error: Extracted text '{span.extracted_text}' not an exact substring of context string '{span.context_string}'."
            elif span.context_string.count(span.extracted_text) != 1:
                agent.state["status"] = "error"
                return f"Error: Extracted text '{span.extracted_text}' is not unique in context string '{span.context_string}'."

            start = text.index(
                span.context_string
            ) + span.context_string.index(span.extracted_text)
            spans.append(
                {
                    "start": start,
                    "end": start + len(span.extracted_text),
                    "text": span.extracted_text,
                }
            )
        agent.state["spans"] = spans
        agent.state["status"] = "success"
        return "Spans extracted successfully."


SYSTEM_TEMPLATE = """
You are an agent that extracts spans from text. The user will provide you with
some text and you will use you SpanAnnotationTool to extract spans from the text.

To extract a span, you should provide the text that you want to extract. However,
since the extracted text may not be unique you should also provide a context string.
This is a string of text that includes extracted text and appears uniquely within the text.

For example, given the text ""The cat chased the dog and the dog chased the cat."
If the task was to extract verbs performed *by the cat* you might return a span with
"chased" as the extracted_text and "the cat chased" as the context_string, to indicate
that you are capturing the *first* instance of "chased" and not the second.

Here is a description of your current task:

{task_description}
"""


class SpanAnnotationAgent(Agent):
    def __init__(
        self,
        task_description,
        *args,
        model=None,
        tools=None,
        system_prompt=None,
        max_steps=3,
        **kwargs,
    ):
        """Init agent."""
        model = model or "gemini/gemini-2.5-flash"
        tools = tools or [SpanAnnotationTool]
        system_prompt = system_prompt or SYSTEM_TEMPLATE.format(
            task_description=task_description
        )
        super().__init__(
            *args,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
            **kwargs,
        )

    def __call__(self, text):
        """Call agent."""
        self.state["status"] = "success"
        self.state["text"] = text
        self.step(text)
        for _ in range(2):
            if self.state["status"] == "success":
                break
            self.step("Please try again.")
        return {
            "status": self.state["status"],
            "spans": self.state.get("spans", []),
        }

    @classmethod
    def apply(cls, task_description, texts, num_workers=1, **kwargs):
        """Apply the agent to a list of texts."""

        def job(text):
            try:
                agent = cls(task_description, **kwargs)
                return agent(text)
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "spans": [],
                }

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(job, text) for text in texts]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
        return [future.result() for future in futures]
