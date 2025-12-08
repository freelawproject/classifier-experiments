from typing import ClassVar

from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from transformers.pipelines.pt_utils import KeyDataset


class Pipeline:
    """Base class for all pipelines."""

    task: str | None = None
    default_pipeline_args: dict = {}
    default_predict_args: dict = {}
    post_process_keys: list[str] = []

    def __init__(self, model: str | dict | None = None, **pipeline_args: dict):
        if self.task is None:
            raise NotImplementedError("`task` must be set.")
        self.model = model
        self.pipeline_args = {**self.default_pipeline_args, **pipeline_args}
        self._pipe = None

    @property
    def pipe(self):
        """Lazy load the pipeline."""
        if self._pipe is None:
            self._pipe = hf_pipeline(model=self.model, **self.pipeline_args)
        return self._pipe

    def predict(self, examples: list, batch_size: int = 1, **kwargs: dict):
        """Run predictions batch predictions."""
        examples = self.prepare_examples(examples)
        preds = []
        dataset = KeyDataset(Dataset.from_dict({"text": examples}), "text")
        post_process_args = {
            k: v for k, v in kwargs.items() if k in self.post_process_keys
        }
        predict_args = {
            "batch_size": batch_size,
            **self.default_predict_args,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in self.post_process_keys
            },
        }
        for out in tqdm(
            self.pipe(dataset, **predict_args),
            desc="Predicting",
            total=len(examples),
        ):
            preds.append(
                self.post_process_prediction(out, **post_process_args)
            )
        return preds

    def prepare_examples(self, examples: list) -> list:
        return examples

    def post_process_prediction(self, prediction: dict) -> dict:
        """Post-process a prediction."""
        return prediction


class ClassificationPipeline(Pipeline):
    """Classification pipeline."""

    task = "classification"
    default_pipeline_args: ClassVar[dict] = {"task": "text-classification"}
    default_predict_args: ClassVar[dict] = {"top_k": None}
    post_process_keys: ClassVar[list[str]] = ["return_scores"]

    def post_process_prediction(
        self, prediction: dict, return_scores: bool = False
    ) -> dict | str:
        """Post-process a prediction."""
        if return_scores:
            return {x["label"]: x["score"] for x in prediction}
        else:
            top_label = max(prediction, key=lambda x: x["score"])
            return top_label["label"]


class TextClassificationPipeline(ClassificationPipeline):
    """Alias for classification pipeline."""

    task = "text-classification"


class MultiLabelClassificationPipeline(ClassificationPipeline):
    """Multi-label classification pipeline."""

    task = "multi-label-classification"

    def post_process_prediction(
        self, prediction: dict, return_scores: bool = False
    ) -> dict | list[str]:
        """Post-process a prediction."""
        if return_scores:
            return {x["label"]: x["score"] for x in prediction}
        else:
            return [x["label"] for x in prediction if x["score"] > 0.5]


class TokenClassificationPipeline(Pipeline):
    """Token classification pipeline."""

    task = "token-classification"
    default_pipeline_args: ClassVar[dict] = {
        "task": "ner",
        "aggregation_strategy": "simple",
    }

    def post_process_prediction(self, prediction: list[dict]) -> list[dict]:
        """Post-process a prediction."""
        for span in prediction:
            span["label"] = span.pop("entity_group")
            span["text"] = span.pop("word")
            if span["text"].startswith(" "):
                span["text"] = span["text"][1:]
                span["start"] += 1
        return prediction


class NERPipeline(TokenClassificationPipeline):
    """Alias for token classification pipeline."""

    task = "ner"
