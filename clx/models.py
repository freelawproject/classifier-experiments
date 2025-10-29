from django.apps import apps

from clx import init_django

__path__: list[str] = []


def __getattr__(name):
    """Load models from django registry."""
    init_django()
    return apps.get_model("app", name)
