from click import group

from .hello import hello
from .manage import manage


@group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(hello)
cli.add_command(manage)
