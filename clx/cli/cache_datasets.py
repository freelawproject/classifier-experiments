import click
from tqdm import tqdm

from clx import pd_save_or_append


@click.command()
def cache_datasets():
    """Dump search datasets to CSV files."""
    from clx.models import Project

    for project in tqdm(Project.objects.all(), desc="Caching datasets"):
        model = project.get_search_model()
        dataset_path = project.cached_documents_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists():
            dataset_path.unlink()
        for data in model.objects.batch_df(
            "id", "text_hash", "text", batch_size=500000
        ):
            pd_save_or_append(data, dataset_path)
