from click import group

from .hello import hello


@group()
def cli():
    """Classifier Experiments CLI."""
    pass


cli.add_command(hello)
