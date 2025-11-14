import ast

import click
import pandas as pd
from tqdm import tqdm

from clx.ml import pipeline
from clx.settings import DATA_DIR
from clx.utils import pd_save_or_append

from .cache_datasets import CACHED_DATASET_DIR

FEATURE_PATH = DATA_DIR / "features" / "scales.csv"

scales2label = {
    "complaint": "Complaint",
    "motion": "Motion",
}


@click.command()
@click.option(
    "do_import",
    "--import",
    is_flag=True,
    help="Import the predictions to docket viewer",
)
@click.option(
    "--reset", is_flag=True, help="Delete cached predictions and re-run"
)
def predict_scales(do_import, reset):
    """Make predictions using the scales model across the docket sample."""
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = []
    if FEATURE_PATH.exists():
        if reset:
            FEATURE_PATH.unlink()
        else:
            existing_ids = pd.read_csv(FEATURE_PATH, usecols=["id"])[
                "id"
            ].tolist()

    if not do_import:
        pipe = pipeline(
            "multi-label-classification",
            model="scales-okn/docket-classification",
            max_length=256,
            truncation=True,
            dtype="float16",
        )
        progress = tqdm(
            desc="Total",
            total=len(
                pd.read_csv(
                    CACHED_DATASET_DIR / "DocketEntry.csv", usecols=["id"]
                )
            ),
        )
        for data in pd.read_csv(
            CACHED_DATASET_DIR / "DocketEntry.csv", chunksize=5000
        ):
            progress.update(len(data))
            data = data[~data["id"].isin(existing_ids)]
            if len(data) > 0:
                data["labels"] = pipe.predict(
                    data["text"].tolist(), batch_size=16
                )
                pd_save_or_append(data, FEATURE_PATH)
    else:
        from clx.models import DocketEntry, Label, LabelFeature

        data = pd.read_csv(FEATURE_PATH, usecols=["id", "labels"])
        data["labels"] = data["labels"].apply(lambda x: ast.literal_eval(x))
        data = data.explode("labels").dropna()
        data = data[data["labels"].isin(scales2label.keys())]

        project = DocketEntry.get_project()
        for scales_label, label_name in tqdm(
            scales2label.items(), desc="Importing features"
        ):
            label, _ = Label.objects.get_or_create(
                project=project, name=label_name
            )
            feature, _ = LabelFeature.objects.get_or_create(
                label=label, name="scales"
            )
            feature_ids = list(
                data[data["labels"] == scales_label]["id"].unique()
            )
            DocketEntry.bulk_replace_feature(feature.id, feature_ids)
