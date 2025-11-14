import click

from clx.settings import DATA_DIR
from clx.utils import pd_save_or_append

CACHED_DATASET_DIR = DATA_DIR / "search_datasets"


@click.command()
def cache_datasets():
    """Make predictions using the scales model across the docket sample."""
    from clx.models import Project

    CACHED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for project in Project.objects.all():
        model_class = project.get_search_model_class()
        dataset_path = CACHED_DATASET_DIR / f"{model_class.__name__}.csv"
        if dataset_path.exists():
            dataset_path.unlink()
        for data in model_class.batch_df("id", "text", batch_size=1000000):
            pd_save_or_append(data, dataset_path)
