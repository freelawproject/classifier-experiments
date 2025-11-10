from typing import ClassVar

import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from .training_run import TrainingRun


class ClassificationRun(TrainingRun):
    """Multi-class classification run."""

    name = "classification"
    add_config_attrs: ClassVar[list[str]] = ["label_names"]
    pipeline_args: ClassVar[dict] = {"task": "text-classification"}
    predict_args: ClassVar[dict] = {"top_k": None}

    def __init__(self, label_names: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = label_names

    def tokenize(self, examples: dict) -> dict:
        """The tokenize function."""
        tokenize_args = {
            "padding": False,
            "truncation": True,
            **self.tokenize_args,
        }
        inputs = self.tokenizer(examples["text"], **tokenize_args)
        inputs["labels"] = torch.tensor(
            [
                self.label_names.index(example_label)
                for example_label in examples["label"]
            ]
        ).long()
        return inputs

    def load_model(self, model_name: str | None = None) -> PreTrainedModel:
        """Load a model for this task."""
        model_name = model_name or self.base_model_name
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_names),
            **self.model_args,
        )
        id2label = dict(enumerate(self.label_names))
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}
        return model

    def compute_metrics(
        self, eval_pred: tuple[torch.Tensor, torch.Tensor]
    ) -> dict:
        """Compute f1 score for evaluation."""
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1).astype(int)
        average = "binary" if len(self.label_names) == 2 else "macro"
        score = f1_score(labels, predictions, average=average)
        return {f"{average}_f1": score}

    def validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate the data format."""
        if "text" not in data.columns:
            raise ValueError("Data must contain a 'text' column.")
        if "label" not in data.columns:
            raise ValueError("Data must contain a 'label' column.")
        if not data["label"].isin(self.label_names).all():
            raise ValueError(
                f"Label column must contain only {self.label_names}."
            )

    def post_process_prediction(self, prediction: dict) -> dict:
        """Post-process a prediction."""
        return {x["label"]: x["score"] for x in prediction}
