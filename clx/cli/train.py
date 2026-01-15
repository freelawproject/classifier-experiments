import click
import pandas as pd

from clx.ml import training_run


@click.command()
@click.argument("project_id")
@click.argument("label_name")
@click.argument("config_name")
def train(project_id, label_name, config_name):
    """Train a label."""
    from clx.models import Label, LabelFinetune, Project

    project = Project.objects.get(id=project_id)
    label = Label.objects.get(project=project, name=label_name)
    train_data, eval_data, run_config = label.prepare_finetune(config_name)

    run = training_run(**run_config)
    outputs = run.train(train_data, eval_data, overwrite=True, remote=True)

    data = pd.concat([train_data, eval_data])

    pipe = label.get_finetune_run_pipe(config_name)
    data["pred"] = pipe(data["text"].tolist(), batch_size=16)
    data = data[data["pred"] == "yes"]

    tag = label.get_trainset_finetune_tag(config_name)
    model = project.get_search_model()
    example_ids = model.objects.filter(
        text_hash__in=data["text_hash"].tolist()
    )
    example_ids = example_ids.values_list("id", flat=True)
    model.bulk_replace_tag(tag.id, example_ids)
    finetune, _ = LabelFinetune.objects.get_or_create(
        label=label, config_name=config_name
    )
    finetune.eval_results = outputs["results"]
    finetune.save()
