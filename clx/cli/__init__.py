from click import group

from .generate_docket_sample import generate_docket_sample
from .manage import manage


@group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(generate_docket_sample)
cli.add_command(manage)
