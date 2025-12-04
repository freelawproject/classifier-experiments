import os
import subprocess

import click
import pandas as pd
from transformers import AutoConfig, AutoModel

from clx.ml import training_run
from clx.settings import CLX_HOME, DATA_DIR

DATASET_PATH = DATA_DIR / "search_datasets" / "docket-entry.csv"
PROJECT_DIR = CLX_HOME / "projects" / "docketbert"


def create_sliced_model(
    name, layers, base_model_name="answerdotai/ModernBERT-base"
):
    out_dir = PROJECT_DIR / "models" / name
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if not out_dir.exists():
        base_model = AutoModel.from_pretrained(base_model_name)
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_hidden_layers = len(layers)
        config.global_attn_every_n_layers = 2
        model = AutoModel.from_config(config)
        base_layers = base_model.layers
        sliced_layers = model.layers
        for new_idx, old_idx in enumerate(layers):
            sliced_layers[new_idx].load_state_dict(
                base_layers[old_idx].state_dict()
            )
        model.embeddings.load_state_dict(base_model.embeddings.state_dict())
        model.save_pretrained(out_dir)
    return out_dir


def get_experiment_config(experiment, batch_size=None):
    config = {
        "task": "mlm",
        "run_dir_parent": PROJECT_DIR / "runs",
        "base_model_name": "answerdotai/ModernBERT-base",
        "tokenizer_name": "answerdotai/ModernBERT-base",
        "tokenize_args": {
            "max_length": 768,
            "padding": False,
        },
        "model_args": {},
        "mlm_probability": 0.3,
        "training_args": {
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "num_train_epochs": 1,
            "warmup_ratio": 0.05,
            "logging_steps": 5,
            "save_strategy": "steps",
            "save_steps": 1000,
            "save_total_limit": 2,
            "eval_strategy": "steps",
            "eval_steps": 1000,
            "prediction_loss_only": True,
            "remove_unused_columns": False,
            "bf16": True,
        },
    }
    default_batch_size = 32

    if experiment == "base-150M":
        default_batch_size = 16
    elif experiment == "large-395M":
        config["base_model_name"] = "answerdotai/ModernBERT-large"
        default_batch_size = 8
    elif experiment == "large-(lr:2e-4)-395M":
        config["training_args"]["learning_rate"] = 2e-4
        config["base_model_name"] = "answerdotai/ModernBERT-large"
        default_batch_size = 8
    elif experiment == "scratch-7M":
        config["model_args"]["config"] = {
            "hidden_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "global_attn_every_n_layers": 2,
        }
    elif experiment == "scratch-16M":
        config["model_args"]["config"] = {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "intermediate_size": 384,
            "global_attn_every_n_layers": 2,
        }
    elif experiment == "scratch-23M":
        config["model_args"]["config"] = {
            "hidden_size": 368,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "global_attn_every_n_layers": 2,
        }
    elif experiment == "scratch-27M":
        config["model_args"]["config"] = {
            "hidden_size": 320,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "intermediate_size": 480,
            "global_attn_every_n_layers": 2,
        }
    elif experiment == "scratch-41M":
        config["model_args"]["config"] = {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "global_attn_every_n_layers": 2,
        }
    elif experiment == "sliced-base-first-4l-59M":
        config["base_model_name"] = create_sliced_model(
            "modernbert-first-4", [0, 1, 2, 3]
        )
    elif experiment == "sliced-base-interleaved-4l-59M":
        config["base_model_name"] = create_sliced_model(
            "modernbert-interleaved-4", [0, 7, 14, 21]
        )
    elif experiment == "sliced-large-first-6l-126M":
        config["base_model_name"] = create_sliced_model(
            "sliced-large-first-6l",
            [0, 1, 2, 3, 4, 5],
            "answerdotai/ModernBERT-large",
        )
    elif experiment == "sliced-large-interleaved-6l-126M":
        config["base_model_name"] = create_sliced_model(
            "sliced-large-interleaved-6l",
            [0, 5, 10, 15, 20, 25],
            "answerdotai/ModernBERT-large",
        )
    elif experiment == "sliced-large-ft-first-6l-126M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-first-6l", [0, 1, 2, 3, 4, 5], base_model_name
        )
    elif experiment == "sliced-large-ft-interleaved-6l-126M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-interleaved-6l",
            [0, 5, 10, 15, 20, 25],
            base_model_name,
        )
    elif experiment == "sliced-large-ft-interleaved-8l-150M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-interleaved-8l",
            [0, 3, 6, 9, 12, 16, 20, 24],
            base_model_name,
        )
    elif experiment == "sliced-large-ft-interleaved-10l-175M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-interleaved-10l",
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            base_model_name,
        )
    elif experiment == "distill-sliced-large-ft-interleaved-10l-175M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-interleaved-10l",
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            base_model_name,
        )
        config["teacher_model_name"] = str(base_model_name)
        config["distill_alpha_ce"] = 0.9
        config["distill_alpha_kl"] = 0.1
        default_batch_size = 8
    elif experiment == "sliced-large-interleaved-10l-175M":
        base_model_name = "answerdotai/ModernBERT-large"
        config["base_model_name"] = create_sliced_model(
            "sliced-large-interleaved-10l",
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            base_model_name,
        )
    elif experiment == "distill-sliced-large-interleaved-10l-175M":
        base_model_name = "answerdotai/ModernBERT-large"
        config["base_model_name"] = create_sliced_model(
            "sliced-large-interleaved-10l",
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            base_model_name,
        )
        config["teacher_model_name"] = str(base_model_name)
        config["distill_alpha_ce"] = 0.9
        config["distill_alpha_kl"] = 0.1
        default_batch_size = 8
    elif experiment == "distill-sliced-large-ft-interleaved-8l-150M":
        base_model_name = (
            PROJECT_DIR / "runs" / "docketbert-large-395M" / "model"
        )
        config["teacher_model_name"] = str(base_model_name)
        config["distill_alpha_ce"] = 0.96
        config["distill_alpha_kl"] = 0.04
        config["base_model_name"] = create_sliced_model(
            "sliced-large-ft-interleaved-8l",
            [0, 3, 6, 9, 12, 16, 20, 24],
            base_model_name,
        )
        default_batch_size = 8
    elif experiment == "distill-base-27M":
        config["teacher_model_name"] = (
            "/workspace/clx/home/runs/docketbert-base-150M/model"
        )
        config["model_args"]["config"] = {
            "hidden_size": 320,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "intermediate_size": 480,
            "global_attn_every_n_layers": 2,
        }
        default_batch_size = 8
    elif experiment == "distill-base-41M":
        config["teacher_model_name"] = (
            "/workspace/clx/home/runs/docketbert-base-150M/model"
        )
        config["model_args"]["config"] = {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "global_attn_every_n_layers": 2,
        }
        default_batch_size = 8
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    if "teacher_model_name" in config:
        config["task"] = "distill-mlm"
        config["teacher_model_args"] = {"dtype": "bfloat16"}

    config["run_name"] = f"docketbert-{experiment}"

    batch_size = batch_size or default_batch_size
    gradient_accumulation_steps = 256 // batch_size
    config["training_args"]["per_device_train_batch_size"] = batch_size
    config["training_args"]["per_device_eval_batch_size"] = batch_size
    config["training_args"]["gradient_accumulation_steps"] = (
        gradient_accumulation_steps
    )
    return config


def kill_pod():
    subprocess.run(
        ["runpodctl", "remove", "pod", os.environ["RUNPOD_POD_ID"]], check=True
    )


@click.command()
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    default=False,
    help="Overwrite the existing run dir.",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    help="Resume training from the last checkpoint.",
)
@click.option(
    "--check-params",
    is_flag=True,
    default=False,
    help="Check parameter count instead of training.",
)
@click.option(
    "--mem-test", is_flag=True, default=False, help="Test memory usage."
)
@click.option(
    "--experiment",
    "-e",
    type=str,
    default="base",
    help="The experiment to run.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=None,
    help="The batch size to use.",
)
@click.option(
    "--exit",
    "-x",
    is_flag=True,
    default=False,
    help="Terminate the runpod pod on finish.",
)
def train_docketbert(
    overwrite, resume, check_params, mem_test, experiment, batch_size, exit
):
    """Train a docket language model."""
    try:
        if resume and overwrite:
            raise click.UsageError(
                "Cannot use --resume and --overwrite together."
            )

        data = pd.read_csv(
            DATASET_PATH,
            usecols=["text"],
            nrows=200000 if mem_test else None,
        )
        data = data.sample(frac=1, random_state=42)
        train_data = data.head(-100000)
        eval_data = data.tail(100000)

        config = get_experiment_config(experiment, batch_size)

        if mem_test:
            config["tokenize_args"]["padding"] = "max_length"

        run = training_run(**config)

        if check_params:
            print(run.load_model().num_parameters())
        else:
            run.train(
                train_data,
                eval_data,
                overwrite=overwrite,
                resume_from_checkpoint=resume,
                lazy_tokenize=True,
            )
    except Exception as e:
        print(e)
        if exit:
            kill_pod()
        raise e
    finally:
        if exit:
            kill_pod()


if __name__ == "__main__":
    train_docketbert()
