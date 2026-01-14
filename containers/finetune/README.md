# Finetune Container

This container enables serverless finetuning for `clx` training runs using RunPod.

## Deployment

To deploy the container, create a new RunPod endpoint using the following configuration:

- Repo: `freelawproject/classifier-experiments`
- Branch: `main`
- Dockerfile Path: `containers/finetune/Dockerfile`
- Build Context: `containers/finetune`

To make sure models persist, add block storage to the endpoint:

`Edit Endpoint -> Advanced -> Network Volumes`

Set the following environment variables for the endpoint:

```bash
CLX_HOME=/runpod-volume/clx
HF_HOME=/runpod-volume/hf
HF_TOKEN=...

# S3 credentials for data transfer
CLX_S3_BUCKET=...
CLX_S3_ACCESS_KEY_ID=...
CLX_S3_SECRET_ACCESS_KEY=...
CLX_S3_REGION=us-east-1
```

These will ensure that all endpoint instances use the shared volume for both `huggingface` and `clx` data. The `HF_TOKEN` is optional, but will allow you to use private models from the Hugging Face Hub.

The S3 credentials are required for the container to download training data from and clean up the S3 bucket.

## Usage

To use serverless finetuning, include the following in your local `.env` file:

```bash
RUNPOD_FINETUNE_ENDPOINT_ID=...
RUNPOD_API_KEY=...
CLX_S3_BUCKET=...
CLX_S3_ACCESS_KEY_ID=...
CLX_S3_SECRET_ACCESS_KEY=...
CLX_S3_REGION=us-east-1
```

The `RUNPOD_FINETUNE_ENDPOINT_ID` is the ID of the RunPod endpoint you created.

You can then use `training_run` as you normally would, with the `remote` argument set to `True` in the `train()` call:

```python
from clx.ml import training_run
import pandas as pd

# Create training run
run = training_run(
    task="classification",
    run_name="sentiment-classifier",
    label_names=["positive", "negative"],
    base_model_name="answerdotai/ModernBERT-large",
)

# Prepare data
train_data = pd.DataFrame({"text": [...], "label": [...]})
eval_data = pd.DataFrame({"text": [...], "label": [...]})

# Finetune remotely
results = run.train(
    train_data,
    eval_data,
    remote=True,
    timeout=7200,  # 2 hour timeout
)

print(results)
# {'status': 'success', 'run_dir': '/runpod-volume/clx/runs/sentiment-classifier', ...}
```

The `train()` method with `remote=True` supports the following additional arguments:

- `endpoint_id`: Override the RunPod endpoint ID (defaults to `RUNPOD_FINETUNE_ENDPOINT_ID` env var).
- `api_key`: Override the RunPod API key (defaults to `RUNPOD_API_KEY` env var).
- `s3_bucket`: Override the S3 bucket (defaults to `CLX_S3_BUCKET` env var).
- `poll_interval`: Seconds between status checks. Defaults to 5.
- `timeout`: Maximum seconds to wait for completion. Defaults to 3600 (1 hour).

## Data Transfer

Training data is transferred via S3:

1. The client uploads `train.csv` and optionally `eval.csv` to `s3://{bucket}/runpod/finetune/{uuid}/`
2. The RunPod handler downloads the data
3. The handler immediately deletes the S3 data (treated as ephemeral)
4. Finetuning proceeds locally on RunPod

This pattern avoids size limits on HTTP POST requests while keeping data transfer secure and temporary.

## Using Finetuned Models

After finetuning completes, the model exists on the RunPod volume. Use `RemotePipeline` for inference:

```python
from clx.ml import pipeline

pipe = pipeline(
    task="classification",
    model=results["run_dir"] + "/model",
    remote=True,
)

predictions = pipe.predict(["I love this!", "I hate this!"])
```
