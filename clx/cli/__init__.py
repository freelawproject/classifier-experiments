import click

from .autopilot import autopilot
from .manage import manage
from .train_multi_label import train_multi_label


@click.group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(manage)
cli.add_command(autopilot)
cli.add_command(train_multi_label)
