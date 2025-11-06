# Classifier Experiments

Classifier Experiments is a toolkit for developing docket classification pipelines.

## Documentation

See the [quickstart](#quickstart) below. For more detail, check out the documentation for the core modules:

* [CLI Commands](clx/cli)
* [Docket Viewer Application](clx/app)
* [Training and Inference Pipelines](clx/ml)
* [LLM Tools](clx/llm)

Some CLI commands generate or pull cached data into your clx home directory. This defaults to `~/clx` and can be configured with the `CLX_HOME` environment variable. See the [Data Branch](https://github.com/freelawproject/classifier-experiments/tree/data) for more details.

## Installation

To install the `clx` package, first clone this repo:

```bash
git clone https://github.com/freelawproject/classifier-experiments
cd classifier-experiments
```

Then you can install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```
or [pip](https://pip.pypa.io/en/stable/getting-started/):

```bash
pip install -e .
```

It is recommended to run `clx config --autoload-env on` after installing the package. See below for more details.

## Configuration

The package can be configured through environment variables or a `.env` file. See [`.env.example`](.env.example) for a complete list of configuration options.

The easiest way to make sure your environment variables are always loaded is to run the following once:

```bash
clx config --autoload-env on
```

This will update your package config to automatically load your `.env` file with `python-dotenv`.

## Quickstart

TODO

```python
# Using models outside of Django
from clx.models import DocketEntry

print(DocketEntry.objects.all().count())
```

## Development

Here are a few tips for setting up your development environment.

### Dev Installation

Install with the `dev` extra to include development dependencies:

```bash
uv sync --extra dev
```
or

```bash
pip install -e '.[dev]'
```

### Pre-commit Hooks

Run the following to install the `pre-commit` hooks:

```bash
pre-commit install
```

Or you can run `pre-commit` manually before committing your changes:

```bash
pre-commit run --all-files
```

### Testing

Run the tests with:

```bash
tox run
```

Or to run a specific test, give the module path as an argument:

```bash
tox run -- tests.test_env.EnvTest.test_env
```

## License

This repository is available under the permissive BSD license, making it easy and safe to incorporate in your own libraries.

## Requirements

- Python 3.13+
