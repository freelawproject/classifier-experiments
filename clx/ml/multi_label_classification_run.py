import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import PreTrainedModel

from .classification_run import ClassificationRun


class MultiLabelClassificationRun(ClassificationRun):
    """Multi-label classification run."""

    task = "multi-label-classification"

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
                [int(label in example_labels) for label in self.label_names]
                for example_labels in examples["labels"]
            ]
        ).float()
        return inputs

    def load_model(self, model_name: str | None = None) -> PreTrainedModel:
        """Load a model for this task."""
        self.model_args["problem_type"] = "multi_label_classification"
        return super().load_model(model_name)

    def compute_metrics(
        self, eval_pred: tuple[torch.Tensor, torch.Tensor]
    ) -> dict:
        """Compute f1 score for evaluation."""
        logits, labels = eval_pred
        logits = torch.sigmoid(torch.Tensor(logits)).numpy()
        predictions = (logits > 0.5).astype(int)
        scores = f1_score(predictions, labels, average=None)
        scores = {
            f"label__{self.label_names[i]}": scores[i]
            for i in range(len(scores))
        }
        scores["f1_macro"] = f1_score(labels, predictions, average="macro")
        return scores

    def validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate the data format."""
        if "text" not in data.columns:
            raise ValueError("Data must contain a 'text' column.")
        if "labels" not in data.columns:
            raise ValueError("Data must contain a 'labels' column.")
        if not data["labels"].apply(lambda x: isinstance(x, list)).all():
            raise ValueError("Labels column must contain a list of labels.")
        if (
            not data["labels"]
            .apply(lambda x: all(label in self.label_names for label in x))
            .all()
        ):
            raise ValueError(
                f"Labels column must contain only lists of {self.label_names}."
            )
