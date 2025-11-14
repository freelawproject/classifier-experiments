import click

from .cache_datasets import cache_datasets
from .config import config
from .generate_docket_sample import generate_docket_sample
from .manage import manage
from .predict_scales import predict_scales


@click.group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(generate_docket_sample)
cli.add_command(manage)
cli.add_command(config)
cli.add_command(predict_scales)
cli.add_command(cache_datasets)
