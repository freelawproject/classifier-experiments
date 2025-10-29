from click import group

from .manage import manage


@group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(manage)
