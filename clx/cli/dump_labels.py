import click
import pandas as pd


@click.command()
def dump_labels():
    """Dump project label data"""
    from clx.models import LabelDecision, Project

    for project in Project.objects.all():
        project.data_dir.mkdir(parents=True, exist_ok=True)
        instructions_path = project.data_dir / "instructions.txt"
        if project.instructions:
            instructions_path.write_text(project.instructions)
        for label in project.labels.all():
            label.data_dir.mkdir(parents=True, exist_ok=True)
            label_instructions_path = label.data_dir / "instructions.txt"
            if label.instructions:
                label_instructions_path.write_text(label.instructions)

            annos_path = label.data_dir / "annos.csv"
            annos = label.load_annos()
            if len(annos):
                annos.to_csv(annos_path, index=False)

            decisions_path = label.data_dir / "decisions.csv"
            decisions = LabelDecision.objects.filter(label=label)
            decisions = pd.DataFrame(
                decisions.values("text_hash", "text", "value", "reason")
            )
            if len(decisions):
                decisions.to_csv(decisions_path, index=False)
