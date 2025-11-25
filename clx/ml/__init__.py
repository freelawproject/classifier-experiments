from pathlib import Path

from .classification_run import ClassificationRun, TextClassificationRun
from .mlm_run import MLMRun
from .multi_label_classification_run import MultiLabelClassificationRun
from .pipelines import (
    ClassificationPipeline,
    MultiLabelClassificationPipeline,
    Pipeline,
    TextClassificationPipeline,
)
from .token_classification_run import NERRun, TokenClassificationRun
from .training_run import TrainingRun

task_registry = [
    ClassificationRun,
    TextClassificationRun,
    MultiLabelClassificationRun,
    TokenClassificationRun,
    NERRun,
    MLMRun,
]


def training_run(
    task: str | None = None,
    load: Path | str | dict | None = None,
    *args,
    **kwargs,
) -> TrainingRun:
    """Get a training run instance from the registry."""
    if load is not None:
        return TrainingRun.load(load)
    if task is None:
        raise ValueError("`task` or `load` must be provided.")
    for task_class in task_registry:
        if task_class.task == task:
            return task_class(*args, **kwargs)
    raise ValueError(
        f"Training run {task} must be one of {[task_class.task for task_class in task_registry]}."
    )


pipeline_registry = [
    ClassificationPipeline,
    TextClassificationPipeline,
    MultiLabelClassificationPipeline,
]


def pipeline(task: str | None = None, *args, **kwargs) -> Pipeline:
    """Get a pipeline instance from the registry."""
    for pipeline_class in pipeline_registry:
        if pipeline_class.task == task:
            return pipeline_class(*args, **kwargs)
    raise ValueError(
        f"Pipeline {task} must be one of {[pipeline_class.task for pipeline_class in pipeline_registry]}."
    )


__all__ = [
    "training_run",
    "task_registry",
    "TrainingRun",
    "ClassificationRun",
    "MultiLabelClassificationRun",
    "Pipeline",
    "ClassificationPipeline",
    "pipeline_registry",
    "pipeline",
]
