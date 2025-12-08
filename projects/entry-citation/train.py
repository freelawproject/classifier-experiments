from utils import PROJECT_DIR, load_train_eval_data

from clx.ml import training_run

run = training_run(
    "ner",
    run_name="entry-citation",
    run_dir_parent=PROJECT_DIR / "runs",
    base_model_name=(
        PROJECT_DIR.parent
        / "docketbert"
        / "runs"
        / "docketbert-sliced-large-ft-interleaved-10l-175M"
        / "model"
    ),
    label_names=["CITATION"],
    training_args={
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 50,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "bf16": True,
    },
    tokenize_args={
        "max_length": 768,
    },
)


if __name__ == "__main__":
    train_data, eval_data = load_train_eval_data()
    run.train(train_data, eval_data, overwrite=True)
