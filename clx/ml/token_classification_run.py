from typing import ClassVar

import pandas as pd
import torch
from transformers import (
    AutoModelForTokenClassification,
    DataCollator,
    DataCollatorForTokenClassification,
    PreTrainedModel,
)

from .training_run import TrainingRun


class TokenClassificationRun(TrainingRun):
    """Token classification run."""

    task = "token-classification"
    add_config_attrs: ClassVar[list[str]] = ["label_names"]

    def __init__(self, label_names: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = label_names
        self._label_map = None

    @property
    def label_map(self):
        """Get BIO-encoded label map."""
        if self._label_map is None:
            label_map = {"O": 0}
            for label_name in sorted(self.label_names):
                label_map["B-" + label_name] = len(label_map)
                label_map["I-" + label_name] = len(label_map)
            self._label_map = label_map
        return self._label_map

    def tokenize(self, examples: dict) -> dict:
        """The tokenize function."""
        tokenize_args = {
            "padding": False,
            "truncation": True,
            "return_offsets_mapping": True,
            **self.tokenize_args,
        }
        inputs = self.tokenizer(examples["text"], **tokenize_args)
        labels = []
        for i in range(len(inputs["input_ids"])):
            input_ids = inputs["input_ids"][i]
            offset_mapping = inputs["offset_mapping"][i]
            spans = examples["spans"][i]
            labels.append(
                self.example_spans_to_labels(input_ids, offset_mapping, spans)
            )
        inputs["labels"] = labels
        del inputs["offset_mapping"]
        return inputs

    def example_spans_to_labels(
        self, input_ids, offset_mapping, spans, o_idx=0
    ):
        """Convert spans to BIO-encoded labels."""
        spans = sorted(spans, key=lambda x: x["start"])

        labels = []
        current_label = None
        for i in range(len(input_ids)):
            offset = offset_mapping[i]

            if len(spans) > 0 and offset[0] >= spans[0]["end"]:
                spans.pop(0)
                current_label = None

            if offset[1] == 0:
                labels.append(-100)
                current_label = None
            elif len(spans) == 0 or offset[1] <= spans[0]["start"]:
                labels.append(o_idx)
                current_label = None
            else:
                if current_label is None:
                    current_label = spans[0]["label"]
                    labels.append(self.label_map["B-" + current_label])
                else:
                    labels.append(self.label_map["I-" + current_label])
        return labels

    def load_data_collator(self) -> DataCollator:
        """Load the data collator."""
        return DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8
        )

    def load_model(self, model_name: str | None = None) -> PreTrainedModel:
        """Load a model for this task."""
        model_name = model_name or self.base_model_name
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_map),
            **self.model_args,
        )
        model.config.label2id = self.label_map
        model.config.id2label = {v: k for k, v in self.label_map.items()}
        return model

    def compute_metrics(
        self, eval_pred: tuple[torch.Tensor, torch.Tensor]
    ) -> dict:
        """Compute metrics method for evaluation."""
        return {}

    def validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate the data format."""
        if "text" not in data.columns:
            raise ValueError("Data must contain a 'text' column.")
        if "spans" not in data.columns:
            raise ValueError("Data must contain a 'spans' column.")
        if not data["spans"].apply(lambda x: isinstance(x, list)).all():
            raise ValueError("Span column must contain a list of spans.")


class NERRun(TokenClassificationRun):
    """Alias for token classification run."""

    task = "ner"
