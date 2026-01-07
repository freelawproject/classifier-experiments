# Inference Container

This container enables serverless inference for `clx` pipelines using RunPod.

## Deployment

To deploy the container, create a new RunPod endpoint using the following configuration:

- Repo: `freelawproject/classifier-experiments`
- Branch: `main`
- Dockerfile Path: `containers/inference/Dockerfile`
- Build Context: `containers/inference`

To make sure models persist, add block storage to the endpoint:

`Edit Endpoint -> Advanced -> Network Volumes`

Finally, set the following environment variables for the endpoint:

```bash
CLX_HOME=/runpod-volume/clx
HF_HOME=/runpod-volume/hf
HF_TOKEN=...
```

These will ensure that all endpoint instances use the shared volume for both `huggingface` and `clx` data. The `HF_TOKEN` is optional, but will allow you to use private models from the Hugging Face Hub.

## Usage

To use serverless inference, include the following in your local `.env` file:

```bash
RUNPOD_INFERENCE_ENDPOINT_ID=...
RUNPOD_API_KEY=...
```

The `RUNPOD_INFERENCE_ENDPOINT_ID` is the ID of the RunPod endpoint you created.

You can then use the `pipeline` loader as you normally would, with the `remote` argument set to `True`:

```python
from clx.ml import pipeline

pipe = pipeline(
    "classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    remote=True,
)

print(pipe.predict(["I love this movie!", "I hate this movie!"], batch_size=2))
```

The `pipe.predict` method for remote pipelines supports the following additional arguments:

- `megabatch_size`: If processing many examples, this is the number that will be processed by each request. Defaults to 1024.
- `num_workers`: Number of megabatches to process in parallel. Defaults to 8.
- `num_retries`: The number of times to retry a prediction if it fails. Defaults to 3.
- `sleep`: The number of seconds to sleep between retries. Defaults to 5.

For example, this will make 2 parallel requests to the endpoint, each with 1 example.

```python
print(pipe.predict(
    ["I love this movie!", "I hate this movie!"],
    megabatch_size=1,
    num_workers=2,
    num_retries=1,
    sleep=1
))
```
