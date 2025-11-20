from typing import ClassVar

import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)

from .training_run import TrainingRun


class MLMRun(TrainingRun):
    """Masked language modeling (MLM) training run."""

    task = "mlm"
    dataset_cols: ClassVar[list[str]] = ["input_ids", "attention_mask"]
    add_config_attrs: ClassVar[list[str]] = ["mlm_probability"]

    def __init__(self, *args, mlm_probability: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_probability = mlm_probability

    def tokenize(self, examples: dict) -> dict:
        """The tokenize function for MLM."""
        tokenize_args = {
            "padding": False,
            "truncation": True,
            **self.tokenize_args,
        }
        inputs = self.tokenizer(examples["text"], **tokenize_args)
        return inputs

    def load_data_collator(self):
        """Use MLM data collator to create masked inputs and labels."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

    def load_model(self, model_name: str | None = None) -> PreTrainedModel:
        """Load a model for MLM, allow custom config to train from scratch."""
        model_name = model_name or self.base_model_name

        model_args = self.model_args.copy()
        config = model_args.pop("config", None)

        if config is not None:
            config = AutoConfig.from_pretrained(model_name, **config)
            return AutoModelForMaskedLM.from_config(config, **model_args)

        return AutoModelForMaskedLM.from_pretrained(
            model_name,
            **model_args,
        )

    def validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate the data format."""
        if "text" not in data.columns:
            raise ValueError("Data must contain a 'text' column.")
