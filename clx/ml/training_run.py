import gc
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import ClassVar

import pandas as pd
import requests
import simplejson as json
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollator,
    DataCollatorWithPadding,
    Pipeline,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from clx.settings import CLX_HOME

DEFAULT_RUN_DIR = CLX_HOME / "runs"


class CSVLoggerCallback(TrainerCallback):
    """Log training metrics to a CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        row = {"step": state.global_step, **logs}
        data = pd.DataFrame([row])

        if self.path.exists():
            existing = pd.read_csv(self.path)
            data = pd.concat([existing, data], ignore_index=True)

        data.to_csv(self.path, index=False)


class TrainingRun:
    """Helper for training runs with Hugging Face models."""

    task: str | None = None
    dataset_cols: ClassVar[list[str]] = [
        "input_ids",
        "attention_mask",
        "labels",
    ]
    add_config_attrs: ClassVar[list[str]] = []
    trainer_class: type[Trainer] = Trainer

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
        self.config_path = self.run_dir / "config.json"

        self._tokenizer = None
        self._base_model = None
        self._model = None
        self._pipe = None

    @property
    def config(self) -> dict:
        """Get the configuration for the training run."""
        config = {
            "task": self.task,
            "run_name": self.run_dir.name,
            "run_dir_parent": str(self.run_dir.parent),
            "base_model_name": str(self.base_model_name),
            "tokenizer_name": str(self.tokenizer_name),
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
            from clx.ml import pipeline

            self._pipe = pipeline(
                task=self.task, model=self.model, tokenizer=self.tokenizer
            )
        return self._pipe

    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        overwrite: bool = False,
        resume_from_checkpoint: str | bool | None = None,
        lazy_tokenize: bool = False,
        callbacks: list[TrainerCallback] | None = None,
        remote: bool = False,
        endpoint_id: str | None = None,
        api_key: str | None = None,
        s3_bucket: str | None = None,
        poll_interval: int = 5,
        timeout: int = 3600,
    ):
        """Run the training run."""
        if remote:
            return self.train_remote(
                train_data=train_data,
                eval_data=eval_data,
                overwrite=overwrite,
                endpoint_id=endpoint_id,
                api_key=api_key,
                s3_bucket=s3_bucket,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Check if the checkpoint directory exists and is not empty
        if self.checkpoint_dir.exists():
            if overwrite:
                shutil.rmtree(self.checkpoint_dir)
            elif not resume_from_checkpoint:
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

            if lazy_tokenize:
                return dataset.with_transform(self.tokenize)

            dataset = dataset.map(self.tokenize, batched=True)
            dataset = dataset.select_columns(self.dataset_cols)
            dataset.set_format(type="torch")
            return dataset

        train_dataset = prepare_dataset(train_data)
        eval_dataset = (
            None if eval_data is None else prepare_dataset(eval_data)
        )

        if callbacks is None:
            callbacks = [CSVLoggerCallback(self.checkpoint_dir / "logs.csv")]

        # Prepare the training arguments
        training_args = {
            "output_dir": self.checkpoint_dir,
            "logging_dir": self.checkpoint_dir / "logs",
            "logging_strategy": "steps",
            "logging_steps": 2,
            "load_best_model_at_end": True,
            "save_total_limit": 2,
            **self.training_args,
        }
        training_args = TrainingArguments(**training_args)

        # Prepare the trainer
        trainer_args = {
            "model": self.base_model,
            "args": training_args,
            "train_dataset": train_dataset,
            "compute_metrics": self.compute_metrics,
            "data_collator": self.load_data_collator(),
            "callbacks": callbacks,
        }
        if eval_dataset is not None:
            trainer_args["eval_dataset"] = eval_dataset

        trainer = self.trainer_class(**trainer_args)
        trainer.training_run = self

        # Write the run config
        self.config_path.write_text(json.dumps(self.config, indent=4))

        # Train the model
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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

    def train_remote(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        overwrite: bool = False,
        endpoint_id: str | None = None,
        api_key: str | None = None,
        s3_bucket: str | None = None,
        poll_interval: int = 5,
        timeout: int = 3600,
    ) -> dict:
        """Submit remote training job to RunPod."""
        from clx.utils import S3

        endpoint_id = endpoint_id or os.getenv("RUNPOD_FINETUNE_ENDPOINT_ID")
        api_key = api_key or os.getenv("RUNPOD_API_KEY")

        if not endpoint_id or not api_key:
            raise ValueError(
                "RUNPOD_FINETUNE_ENDPOINT_ID and RUNPOD_API_KEY must be set"
            )

        s3 = S3(bucket=s3_bucket)
        job_key = str(uuid.uuid4())
        s3_prefix = f"runpod/finetune/{job_key}"

        # Upload training data
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            train_data.to_csv(train_path, index=False)
            s3.upload(train_path, f"{s3_prefix}/train.csv")

            if eval_data is not None:
                eval_path = Path(tmpdir) / "eval.csv"
                eval_data.to_csv(eval_path, index=False)
                s3.upload(eval_path, f"{s3_prefix}/eval.csv")

        # Build payload
        config = self.config
        del config["run_dir_parent"]
        payload = {
            "input": {
                "training_run": config,
                "s3_bucket": s3.bucket,
                "s3_prefix": s3_prefix,
                "overwrite": overwrite,
            }
        }

        # Submit job
        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/run",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        job_id = response.json()["id"]

        # Check progress
        start_time = time.time()
        pbar = None

        try:
            while 1:
                status_response = requests.get(
                    f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                status_response.raise_for_status()
                status_data = status_response.json()

                progress = status_data.get("output", {})
                if progress and "max_steps" in progress:
                    if pbar is None:
                        pbar = tqdm(
                            total=progress["max_steps"],
                            desc="Finetuning",
                            unit="step",
                        )
                    pbar.n = progress.get("step", 0)
                    pbar.refresh()

                if status_data["status"] == "COMPLETED":
                    if pbar is not None:
                        pbar.n = pbar.total
                        pbar.refresh()
                    return status_data["output"]
                elif status_data["status"] in ("FAILED", "CANCELLED"):
                    raise RuntimeError(
                        f"Training job failed: {status_data.get('error', 'Unknown error')}"
                    )

                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Training job timed out after {timeout} seconds"
                    )

                time.sleep(poll_interval)
        finally:
            if pbar is not None:
                pbar.close()

    def predict(self, *args, **kwargs) -> list[dict]:
        """Run predictions with pipeline."""
        return self.pipe.predict(*args, **kwargs)

    @classmethod
    def load(cls, path_or_config: Path | str | dict) -> "TrainingRun":
        """Load a training run from a config."""
        from clx.ml import training_run

        if isinstance(path_or_config, dict):
            config = path_or_config
        else:
            config_path = Path(path_or_config)
            if config_path.name != "config.json":
                config_path = config_path / "config.json"
            if not config_path.exists():
                config_path = DEFAULT_RUN_DIR / config_path
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Training run config {path_or_config} does not exist. "
                        "Please create it or provide a config."
                    )
            config = json.loads(config_path.read_text())
        task = config.pop("task")
        return training_run(task=task, **config)
