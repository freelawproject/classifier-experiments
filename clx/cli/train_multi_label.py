import click
import pandas as pd
from tqdm import tqdm

from clx.ml import pipeline, training_run
from clx.settings import CLX_HOME
from clx.utils import pd_save_or_append


def get_label_predictions(data_dir, data, label):
    preds_path = data_dir / "preds" / f"{label.id}.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    needs_preds = data.copy()

    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        needs_preds = needs_preds[
            ~needs_preds["document_id"].isin(preds["document_id"])
        ]

    if len(needs_preds) > 0:
        needs_preds["score"] = label.pipe.predict(
            needs_preds["text"].tolist(),
            num_workers=32,
            return_scores=True,
        )
        needs_preds["score"] = needs_preds["score"].apply(lambda x: x["yes"])
        needs_preds = needs_preds[["document_id", "score"]]
        pd_save_or_append(needs_preds, preds_path)
    return pd.read_csv(preds_path)


def train(data, project_id, label_ids, remote=False):
    data = data.copy()[["text", "labels"]]

    data = data.sample(frac=1)
    split = int(0.8 * len(data))
    train_data = data.head(split)
    eval_data = data.tail(-split)

    batch_size = 8
    gradient_accumulation_steps = 1
    num_train_epochs = 3

    save_steps = (
        (len(train_data) / (batch_size * gradient_accumulation_steps))
        * num_train_epochs
    ) // 10
    eval_steps = save_steps

    train_run = training_run(
        task="multi-label-classification",
        run_name=f"multi-label-{project_id}",
        label_names=label_ids,
        # base_model_name="docketanalyzer/modernbert-unit-test",
        tokenize_args={
            "max_length": 768,
            "padding": False,
        },
        training_args={
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "bf16": True,
        },
    )
    train_run.train(train_data, eval_data, overwrite=True, remote=remote)


@click.command()
@click.argument("project_name", type=str)
@click.option("--remote", is_flag=True, default=False)
def train_multi_label(project_name, remote):
    from clx.models import LabelDocument, Project

    project = Project.objects.get(name=project_name)
    data_dir = project.project_dir / "train"
    data_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / "data.csv"
    if not path.exists():
        data = LabelDocument.objects.filter(label__project=project)
        data = data.values_list("document__id", "document__text")
        data = pd.DataFrame(data, columns=["document_id", "text"])
        data = data.drop_duplicates(subset=["document_id"])
        data = data.sample(frac=1)
        data.to_csv(path, index=False)
    else:
        data = pd.read_csv(path)

    label2name = {}
    for label in tqdm(project.labels.all(), desc="Getting label predictions"):
        if label.finetune_status == "completed":
            label2name[label.id] = label.name
            label_data = get_label_predictions(data_dir, data, label)
            label_data = label_data.rename(columns={"score": str(label.id)})
            data = data.merge(label_data, on="document_id", how="left").fillna(
                0
            )
    label_ids = list(label2name.keys())

    data["labels"] = data[label_ids].apply(
        lambda x: [lid for lid in label_ids if x[lid] > 0.5], axis=1
    )

    model_path = CLX_HOME / "runs" / f"multi-label-{project.id}" / "model"

    if not model_path.exists():
        train(data, project.id, label_ids, remote)

    pipe = pipeline(
        "multi-label-classification",
        model=model_path,
        remote=remote,
        max_length=768,
        bf16=True,
    )
    data["predictions"] = pipe.predict(
        data["text"].tolist(), return_scores=True, batch_size=8
    )
    data["predicted_labels"] = data["predictions"].apply(
        lambda x: [label2name[lid] for lid in x if x[lid] > 0.5]
    )
    data.to_csv(data_dir / "predictions.csv", index=False)
