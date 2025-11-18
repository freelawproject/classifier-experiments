from django.apps import AppConfig
from django.db.models.signals import post_migrate


class AppConfig(AppConfig):
    name = "clx.app"

    def ready(self):
        from .search_utils import init_search_models

        post_migrate.connect(init_search_models)
