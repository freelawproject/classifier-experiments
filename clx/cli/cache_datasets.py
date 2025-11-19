import click
from tqdm import tqdm

from clx import pd_save_or_append
from clx.settings import CACHED_DATASET_DIR


@click.command()
def cache_datasets():
    """Dump search datasets to CSV files."""
    from clx.models import Project

    CACHED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for project in tqdm(Project.objects.all(), desc="Caching datasets"):
        model = project.get_search_model()
        dataset_path = CACHED_DATASET_DIR / f"{project.id}.csv"
        if dataset_path.exists():
            dataset_path.unlink()
        for data in model.objects.batch_df("id", "text", batch_size=500000):
            pd_save_or_append(data, dataset_path)
