# Classifier Experiments

Classifier Experiments is a toolkit for developing docket classification pipelines.

## Documentation

See the [quickstart](#quickstart) below. For more detail, check out the documentation for the core modules:

* [CLI Commands](clx/cli)
* [Docket Viewer Application](clx/app)
* [Training and Inference Pipelines](clx/ml)
* [LLM Tools](clx/llm)

## Installation

To install the `clx` package, first clone this repo:

```bash
git clone https://github.com/freelawproject/classifier-experiments
cd classifier-experiments
```

Then you can install with [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://pip.pypa.io/en/stable/getting-started/):

* Using `uv`:
    ```bash
    uv sync
    ```
    Use `uv sync --extra dev` to install development dependencies.

* Using `pip`:
   ```bash
   pip install -e .
   ```
   Use `pip install -e '.[dev]'` to install development dependencies.

## Configuration

The package can be configured through environment variables or a `.env` file. See [`.env.example`](.env.example) for a complete list of configuration options.

## Quickstart

TODO

```python
import clx
```

## Development

Be sure to run `pre-commit` before committing your changes.

```bash
pre-commit install
```

Or you can run `pre-commit` manually before committing your changes:

```bash
pre-commit run --all-files
```

## License

This repository is available under the permissive BSD license, making it easy and safe to incorporate in your own libraries.

## Requirements

- Python 3.13+
