import click
import pandas as pd
import simplejson as json

from clx import label2slug
from clx.ml import training_run
from clx.settings import CLX_HOME


@click.command()
@click.argument("project_id")
@click.argument("label_name")
@click.argument("config_name")
@click.option(
    "do_import",
    "--import",
    is_flag=True,
    default=False,
    help="Export the trainset to a CSV file.",
)
@click.option(
    "do_export",
    "--export",
    is_flag=True,
    default=False,
    help="Export the trainset to a CSV file.",
)
def train(project_id, label_name, config_name, do_import, do_export):
    """Train a label."""
    finetune_dir = (
        CLX_HOME
        / "app_projects"
        / project_id
        / "labels"
        / f"{label2slug(label_name)}"
        / config_name
    )
    config_path = finetune_dir / "config.json"
    train_path = finetune_dir / "train.csv"
    eval_path = finetune_dir / "eval.csv"
    pred_path = finetune_dir / "preds.csv"
    finetune_dir.mkdir(parents=True, exist_ok=True)

    if do_export:
        from clx.models import Label, Project

        project = Project.objects.get(id=project_id)
        label = Label.objects.get(project=project, name=label_name)
        train_data, eval_data, run_config = label.prepare_finetune(config_name)
        train_data.to_csv(train_path, index=False)
        eval_data.to_csv(eval_path, index=False)
        config_path.write_text(json.dumps(run_config, indent=4))
    elif do_import:
        from clx.models import Label, LabelFinetune, Project

        project = Project.objects.get(id=project_id)
        label = Label.objects.get(project=project, name=label_name)

        tag = label.get_trainset_finetune_tag(config_name)
        data = pd.read_csv(pred_path)
        data = data[data["pred"]]
        model = project.get_search_model()
        example_ids = model.objects.filter(
            text_hash__in=data["text_hash"].tolist()
        )
        example_ids = example_ids.values_list("id", flat=True)
        model.bulk_replace_tag(tag.id, example_ids)
        finetune, _ = LabelFinetune.objects.get_or_create(
            label=label, config_name=config_name
        )
        eval_results = json.loads(
            (finetune_dir / "run" / "checkpoints" / "results.json").read_text()
        )
        finetune.eval_results = eval_results
        finetune.save()
    else:
        train_data = pd.read_csv(train_path)
        eval_data = pd.read_csv(eval_path)
        run_config = json.loads(config_path.read_text())
        run_config["run_name"] = "run"
        run_config["run_dir_parent"] = str(finetune_dir)
        run = training_run(**run_config)
        run.train(train_data, eval_data, overwrite=True)

        data = pd.concat([train_data, eval_data])
        data["pred"] = run.predict(data["text"].tolist(), batch_size=16)
        data["label"] = data["label"] == "yes"
        data["pred"] = data["pred"] == "yes"
        data["agree"] = data["pred"] == data["label"]
        data.to_csv(pred_path, index=False)
