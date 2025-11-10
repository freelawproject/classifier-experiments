import json
from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


class DSPyPredictor:
    default_model = "bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
    default_signature_str = "text: str -> value: bool"
    default_input_fields: ClassVar[list[str]] = ["text"]
    default_optimizer_args: ClassVar[dict] = {}

    def __init__(
        self,
        model: str | dict | None = None,
        signature_str: str | None = None,
        instructions: str | None = None,
        input_fields: list[str] | None = None,
        optimizer_args: dict | None = None,
    ):
        """Initialize the DSPy program."""
        model = model or self.default_model
        if isinstance(model, str):
            model = {"model": model, "temperature": 1.0, "max_tokens": 16000}
        self.model = model
        self.signature_str = signature_str or self.default_signature_str
        self.instructions = instructions
        self.input_fields = input_fields or self.default_input_fields
        self.optimizer_args = optimizer_args or self.default_optimizer_args
        self._program = None

    @property
    def config(self):
        return {
            "model": self.model,
            "signature_str": self.signature_str,
            "instructions": self.instructions,
            "input_fields": self.input_fields,
            "optimizer_args": self.optimizer_args,
            "state": self.program.dump_state(),
        }

    def save(self, path: Path | str):
        Path(path).write_text(json.dumps(self.config, indent=4))

    @classmethod
    def from_config(cls, config: dict | str | Path):
        if isinstance(config, str):
            config = Path(config)
        if isinstance(config, Path):
            config = json.loads(config.read_text())
        state = config.pop("state")
        program = cls(**config)
        if state:
            program.program.load_state(state)
        return program

    def create_program(self):
        signature = dspy.Signature(
            self.signature_str, instructions=self.instructions
        )
        return dspy.Predict(signature)

    @property
    def program(self):
        if self._program is None:
            self._program = self.create_program()
        return self._program

    def prepare_examples(self, examples: list[dict | str | dspy.Example]):
        prepared_examples = []
        for example in examples:
            if isinstance(example, str):
                example = {"text": example}
            if isinstance(example, dict):
                example = dspy.Example(**example)
            prepared_examples.append(example.with_inputs(*self.input_fields))
        return prepared_examples

    def predict(
        self,
        examples: list[dict | str | dspy.Example],
        num_workers: int | None = None,
    ):
        lm = dspy.LM(**self.model)
        with dspy.context(lm=lm):
            return self.program.batch(
                self.prepare_examples(examples),
                num_threads=num_workers,
            )

    def load_optimizer(self, num_workers: int | None = None):
        def metric(e, p, *args, **kwargs):
            return int(bool(e.value) == bool(p.value))

        optimizer_args = {
            "metric": metric,
            "auto": "light",
            **self.optimizer_args,
            "num_threads": num_workers,
        }
        return dspy.MIPROv2(**optimizer_args)

    def fit(
        self,
        examples: list[dict | str | dspy.Example],
        num_workers: int | None = None,
    ):
        lm = dspy.LM(**self.model)
        with dspy.context(lm=lm):
            examples = self.prepare_examples(examples)
            optimizer = self.load_optimizer(num_workers)
            self._program = optimizer.compile(self.program, trainset=examples)


class GEPAPredictor(DSPyPredictor):
    default_signature_str = "text: str -> value: bool, reason: str"
    default_optimizer_args: ClassVar[dict] = {
        "reflection_lm": {
            "model": "bedrock/us-west-2.claude-sonnet-4-5-20250929-v1:0/",
            "temperature": 1.0,
            "max_tokens": 32000,
        },
    }

    def load_optimizer(self, num_workers: int | None = None):
        def metric(e, p, *args, **kwargs):
            result = {"score": int(bool(e.value) == bool(p.value))}
            if e.reason:
                result["feedback"] = e.reason
            return ScoreWithFeedback(**result)

        optimizer_args = deepcopy(self.optimizer_args)
        reflection_lm = deepcopy(self.default_optimizer_args["reflection_lm"])
        if "reflection_lm" in optimizer_args:
            reflection_lm = optimizer_args.pop("reflection_lm")
        optimizer_args = {
            "metric": metric,
            "auto": "light",
            **optimizer_args,
            "reflection_lm": dspy.LM(**reflection_lm),
            "num_threads": num_workers,
        }
        return dspy.GEPA(**optimizer_args)
