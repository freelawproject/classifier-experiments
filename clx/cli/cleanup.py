import click
from tqdm import tqdm


@click.command()
def cleanup():
    """Sync app data."""
    from clx.models import LabelHeuristic, Project

    # Guarantee tags rows for all projects.
    print("Guaranteeing tags rows for all projects...")
    for project in Project.objects.all():
        project.get_search_model().guarantee_tags_rows()

    # Sync custom heuristics.
    print("Syncing custom heuristics...")
    LabelHeuristic.sync_custom_heuristics()

    # Apply new heuristics
    for heuristic in tqdm(
        list(LabelHeuristic.objects.filter(applied_at__isnull=True))
    ):
        print(f"Applying heuristic {heuristic.name}...")
        heuristic.apply()

    # Update out of date heuristics.
    for project in Project.objects.all():
        search_model = project.get_search_model()
        last_updated = (
            search_model.objects.order_by("-updated_at").first().updated_at
        )
        for heuristic in LabelHeuristic.objects.filter(label__project=project):
            if (
                heuristic.applied_at is not None
                and heuristic.applied_at < last_updated
            ):
                print(f"Updating heuristic {heuristic.name}...")
                heuristic.apply()
