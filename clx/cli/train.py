import click


@click.command()
@click.argument("project_id")
@click.argument("label_name")
@click.argument("config_name")
def train(project_id, label_name, config_name):
    """Train a finetune model for a label."""
    from clx.models import Label

    label = Label.objects.get(project_id=project_id, name=label_name)
    label.train_finetune(config_name)
