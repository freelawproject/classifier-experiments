# Training and Inference Pipelines

## Training Runs

We use `TrainingRun`s to standardize the training, evaluation, and inference pipelines for various tasks. Each `TrainingRun` instance is registered with a `task` name. You can train a model for a given task as follows:

```python
from clx.ml import training_run

run = training_run("classification", "my-run-name")
run.train(train_data, eval_data, overwrite=True)
```

By default, the training run will save data in `LOCAL_DATA_DIR / "runs" / "my-run-name"`. This can be customized by passing a `run_dir_parent` argument. You can load a training run instance from a run directory as follows:

```python
from clx.ml import training_run

run = training_run(load="path/to/run-dir")
```

If you trained a model, you can load it for inference as follows:

```python
from clx.ml import training_run

run = training_run(load="path/to/run-dir")
preds = run.predict(texts)
```

Here are some arguments you can pass to customize your training run:

- `run_name`: The name of the training run.
- `run_dir_parent`: The parent directory of the training run.
- `base_model_name`: The Hugging Face model name of the base model to finetune.
- `tokenizer_name`: The name of the tokenizer to use (defaults to the base model name).
- `tokenize_args`: Passed to the tokenize function when preparing your dataset.
- `model_args`: Passed to corresponding HF `AutoModel` class for your task.
- `training_args`: Passed to the HF `TrainingArguments`. See the [HF documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.TrainingArguments) for the full set of options.

Some tasks may require additional arguments. For example, the `classification` task requires a `label_names` argument.

The `train_data` and optional `eval_data` sent to `run.train` should be a pandas DataFrame with a task-specific format. See the task-specific documentation below for details.

### Tasks

#### Classification

**Data Format:**

| text | label |
|------|-----|
| "apple" | "fruit" |
| "banana" | "fruit" |
| "cat" | "animal" |

**Example:**

```python
import pandas as pd

data = pd.DataFrame({
    "text": ["apple", "banana", "cat"],
    "label": ["fruit", "fruit", "animal"],
})
```

**Training:**

Initialization requires a list of `label_names`.

```python
import pandas as pd
from clx.ml import training_run

data = pd.DataFrame({
    "text": ["apple", "banana", "cat"],
    "label": ["fruit", "fruit", "animal"],
})
run = training_run("classification", "my-run-name", label_names=["fruit", "animal"])
run.train(data)
```

**Inference:**

Predict outputs a dictionary mapping label names to scores for each example.

```python
run = training_run(load=LOCAL_DATA_DIR / "runs" / "my-run-name")
preds = run.predict(data["text"].tolist(), batch_size=8)
print(preds)

"""
Output:
[{'fruit': 0.95, 'animal': 0.05}, {'fruit': 0.85, 'animal': 0.15}, {'fruit': 0.25, 'animal': 0.75}]
"""
```

#### Multi-label Classification

TODO

### Creating a new Training Run

To create a new training run, extend the `TrainingRun` class and register it with the `task_registry` in the `training_runs/__init__.py` file. Here is a template of the various ways you can customize your training run:

```python
from clx.ml import TrainingRun

class MyTrainingRun(TrainingRun):
    """My training run."""
    # Required: The name of the training run.
    name = ...

    # Required: The columns to select from the Dataset object when converting to Tensors. These should match the inputs expected by your `AutoModel` class.
    dataset_cols = ["input_ids", "attention_mask", ...]

    # Optional: If you add any additional arguments to __init__, add them here so they are dumped to the run config (must be serializable).
    add_config_attrs = ["my-config-attr"]

    # Provide pipeline initialization arguments to the HF pipeline constructor for inference.
    pipeline_args = {"task": "hf-pipeline-task-name", ...}

    # Optional: Default arguments to pass to `run.pipe` for inference.
    predict_args = {...}

    # Only add this if you want to use a custom Trainer class e.g. for custom loss functions.
    trainer_class = ...

    def validate_data_format(self, data: pd.DataFrame) -> None:
        # You must implement this to define your task-specific data format. Raise errors if the user's data is not in the expected format.

    def load_model(self, model_name: str | None = None) -> PreTrainedModel:
        # Return an instance of the appropriate `AutoModel` class for your task. You should pass `self.model_args` to the `from_pretrained` method to allow user overrides.

    def tokenize(self, examples: dict) -> dict:
        # Here is the base implementation for the tokenize function that is mapped over your dataset. Override this if you need to prepare any task-specific inputs.
        return self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            **self.tokenize_args,
        )

    def compute_metrics(self, eval_pred: tuple[torch.Tensor, torch.Tensor]) -> dict:
        # Implement this for your `compute_metrics` method used by the Trainer.

    def post_process_prediction(self, prediction: dict) -> dict:
        # Implement this to reformat the prediction returned by the Hugging Face pipeline.
```

See the `TrainingRun` implementation for additional hooks and overrides, however these are the main ones you will typically need for a new task.
