from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from tqdm import tqdm


@click.command()
@click.option("--update", is_flag=True, help="Update labels")
@click.option("--predict", is_flag=True, help="Run global corpus predictions")
@click.option("--label-name", "--label", help="Label name")
def cleanup(update, predict, label_name):
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
        last_created_example = search_model.objects.order_by(
            "-created_at"
        ).first()
        if last_created_example is not None:
            for heuristic in LabelHeuristic.objects.filter(
                label__project=project
            ):
                if (
                    heuristic.applied_at is not None
                    and heuristic.applied_at < last_created_example.created_at
                ):
                    print(f"Updating heuristic {heuristic.name}...")
                    heuristic.apply()

        def update_label(label, predict):
            print(f"Updating label {label.name}...")
            label.update_all(predict=predict, num_threads=32)

        if update:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for label in project.labels.all().order_by("name"):
                    if label_name is None or label.name == label_name:
                        futures.append(
                            executor.submit(update_label, label, predict)
                        )
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass
