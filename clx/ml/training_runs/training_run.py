import gc
import json
import shutil
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollator,
    DataCollatorWithPadding,
    Pipeline,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

from clx.settings import LOCAL_DATA_DIR

DEFAULT_RUN_DIR = LOCAL_DATA_DIR / "runs"


class TrainingRun:
    """Helper for training runs with Hugging Face models."""

    name: str | None = None
    dataset_cols: ClassVar[list[str]] = [
        "input_ids",
        "attention_mask",
        "labels",
    ]
    add_config_attrs: ClassVar[list[str]] = []
    trainer_class: type[Trainer] = Trainer
    pipeline_args: ClassVar[dict] = {}
    predict_args: ClassVar[dict] = {}

    def __init__(
        self,
        run_name: str,
        run_dir_parent: Path | str = DEFAULT_RUN_DIR,
        base_model_name: str = "answerdotai/ModernBERT-large",
        tokenizer_name: str | None = None,
        tokenize_args: dict | None = None,
        model_args: dict | None = None,
        training_args: dict | None = None,
    ):
        self.run_dir = Path(run_dir_parent) / run_name
        self.base_model_name = base_model_name
        self.tokenizer_name = tokenizer_name or base_model_name
        self.tokenize_args = tokenize_args or {}
        self.model_args = model_args or {}
        self.training_args = training_args or {}

        self.model_dir = self.run_dir / "model"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.results_path = self.checkpoint_dir / "results.json"
        self.logging_dir = self.checkpoint_dir / "logs"
        self.config_path = self.run_dir / "config.json"

        self._tokenizer = None
        self._base_model = None
        self._model = None
        self._pipe = None

    @property
    def config(self) -> dict:
        """Get the configuration for the training run."""
        config = {
            "task": self.name,
            "run_name": self.run_dir.name,
            "run_dir_parent": str(self.run_dir.parent),
            "base_model_name": self.base_model_name,
            "tokenizer_name": self.tokenizer_name,
            "tokenize_args": self.tokenize_args,
            "model_args": self.model_args,
            "training_args": self.training_args,
        }
        for attr in self.add_config_attrs:
            config[attr] = getattr(self, attr)
        return config

    def validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate the data format."""
        raise NotImplementedError("Subclasses must implement this method.")

    def load_model(
        self, model_name: str | Path | None = None
    ) -> PreTrainedModel:
        """Load a model for this task."""
        model_name = model_name or self.base_model_name
        model = AutoModel.from_pretrained(model_name, **self.model_args)
        return model

    def tokenize(self, examples: dict) -> dict:
        """The tokenize function."""
        inputs = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            **self.tokenize_args,
        )
        return inputs

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_data_collator(self) -> DataCollator:
        """Load the data collator."""
        return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

    def compute_metrics(
        self, eval_pred: tuple[torch.Tensor, torch.Tensor]
    ) -> dict:
        """Compute metrics method for evaluation."""
        return {}

    def load_pipe(self, **pipeline_args: dict) -> Pipeline:
        """Load the pipeline."""
        return pipeline(**self.pipeline_args, **pipeline_args)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    @property
    def base_model(self) -> PreTrainedModel:
        """Get the base model."""
        if self._base_model is None:
            self._base_model = self.load_model()
        return self._base_model

    @property
    def model(self) -> PreTrainedModel:
        """Get the model."""
        if self._model is None:
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Model directory {self.model_dir} does not exist.",
                    "Please train the model first with `run.train`.",
                )
            self._model = self.load_model(self.model_dir)
        return self._model

    @property
    def pipe(self) -> Pipeline:
        """Inference pipeline for trained model."""
        if self._pipe is None:
            self._pipe = self.load_pipe(
                model=self.model,
                tokenizer=self.tokenizer,
            )
        return self._pipe

    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        overwrite: bool = False,
    ):
        """Run the training run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Check if the checkpoint directory exists and is not empty
        if self.checkpoint_dir.exists():
            if overwrite:
                shutil.rmtree(self.checkpoint_dir)
            else:
                raise FileExistsError(
                    f"Checkpoint directory {self.checkpoint_dir} already exists.",
                    "Please delete it or set `overwrite=True` to overwrite it.",
                )

        # Validate the input data format
        self.validate_data_format(train_data)
        if eval_data is not None:
            self.validate_data_format(eval_data)

        # Prepare the datasets
        def prepare_dataset(data: pd.DataFrame) -> Dataset:
            dataset = Dataset.from_pandas(data)
            dataset = dataset.map(self.tokenize, batched=True)
            dataset = dataset.select_columns(self.dataset_cols)
            dataset.set_format(type="torch")
            return dataset

        train_dataset = prepare_dataset(train_data)
        eval_dataset = (
            None if eval_data is None else prepare_dataset(eval_data)
        )

        # Prepare the training arguments
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            logging_dir=self.logging_dir,
            logging_strategy="steps",
            logging_steps=2,
            **self.training_args,
        )

        # Prepare the trainer
        trainer_args = {
            "model": self.base_model,
            "args": training_args,
            "train_dataset": train_dataset,
            "compute_metrics": self.compute_metrics,
            "data_collator": self.load_data_collator(),
        }
        if eval_dataset is not None:
            trainer_args["eval_dataset"] = eval_dataset

        trainer = self.trainer_class(**trainer_args)

        # Write the run config
        self.config_path.write_text(json.dumps(self.config, indent=4))

        # Train the model
        trainer.train()

        # Evaluate the model
        if eval_dataset is not None:
            eval_results = trainer.evaluate(eval_dataset)
            eval_results["num_train_examples"] = len(train_data)
            eval_results["num_eval_examples"] = len(eval_data)
            self.results_path.write_text(json.dumps(eval_results, indent=4))

        # Save the model
        trainer.save_model(self.model_dir)

        # Save the tokenizer
        self.tokenizer.save_pretrained(self.model_dir)

        # Clear the GPU
        self._base_model = None
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def post_process_prediction(self, prediction: dict) -> dict:
        """Post-process a prediction."""
        return prediction

    def predict(
        self, texts: str | list[str], batch_size: int = 1, **predict_args: dict
    ) -> list[dict]:
        """Run predictions batch predictions."""
        preds = []
        dataset = KeyDataset(Dataset.from_dict({"text": texts}), "text")
        for out in self.pipe(
            dataset, batch_size=batch_size, **self.predict_args, **predict_args
        ):
            preds.append(self.post_process_prediction(out))
        return preds

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "TrainingRun":
        """Load a training run from a config."""
        from clx.ml import training_run

        run_dir = Path(run_dir)
        config = json.loads((run_dir / "config.json").read_text())
        task = config.pop("task")
        return training_run(task=task, **config)
