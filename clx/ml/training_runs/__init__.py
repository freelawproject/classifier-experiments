from pathlib import Path

from .classification_run import ClassificationRun
from .multi_label_classification_run import MultiLabelClassificationRun
from .training_run import TrainingRun

task_registry = [
    ClassificationRun,
    MultiLabelClassificationRun,
]


def training_run(
    task: str | None = None, load: Path | str | None = None, *args, **kwargs
) -> TrainingRun:
    """Get a training run instance from the registry."""
    if load is not None:
        return TrainingRun.from_run_dir(load)
    if task is None:
        raise ValueError("`task` or `load` must be provided.")
    for task_class in task_registry:
        if task_class.name == task:
            return task_class(*args, **kwargs)
    raise ValueError(
        f"Training run {task} must be one of {[task_class.name for task_class in task_registry]}."
    )


__all__ = [
    "training_run",
    "task_registry",
    "TrainingRun",
    "ClassificationRun",
    "MultiLabelClassificationRun",
]
