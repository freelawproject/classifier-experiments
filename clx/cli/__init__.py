import click

from .config import config
from .generate_docket_sample import generate_docket_sample
from .manage import manage


@click.group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(generate_docket_sample)
cli.add_command(manage)
cli.add_command(config)
