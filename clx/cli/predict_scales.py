import click
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from clx import label2slug
from clx.ml import pipeline
from clx.settings import CLX_HOME
from clx.utils import pd_save_or_append

SCALES_DIR = CLX_HOME / "app_projects" / "docket-entry" / "scales"
MODEL_DIR = CLX_HOME / "models" / "scales"

scales2label = {
    "summons": "Summons",
    "waiver": "Waiver",
    "brief": "Brief / Memorandum",
    "arrest": "Arrest",
    "warrant": "Warrant",
    "verdict": "Verdict",
    "answer": "Answer",
    # "complaint": "Complaint",
    "indictment": "Indictment",
    "information": "Information",
    "petition": "Petition",
    "notice": "Notice",
    "response": "Reply / Response",
    "minute entry": "Minute Entry",
    "plea agreement": "Plea Agreement",
    "judgment": "Judgment",
    "stipulation": "Stipulation",
    "motion": "Motion / Request",
    "order": "Order",
}


@click.command()
@click.option(
    "do_import",
    "--import",
    is_flag=True,
    help="Import the predictions to docket viewer",
)
def predict_scales(do_import):
    """Make predictions using the scales model across the docket sample."""
    SCALES_DIR.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_csv(
        CLX_HOME / "app_projects" / "docket-entry" / "docs.csv"
    )

    if not do_import:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        all_data = all_data.drop_duplicates(subset=["text_hash"])

        for scales_label, label_name in tqdm(
            scales2label.items(), desc="Predicting scales"
        ):
            scales_path = SCALES_DIR / f"{label2slug(label_name)}.csv"

            data = all_data.copy()
            if scales_path.exists():
                existing_data = pd.read_csv(scales_path, usecols=["text_hash"])
                data = data[
                    ~data["text_hash"].isin(existing_data["text_hash"])
                ]

            if len(data) != 0:
                model_name = (
                    f"scales-okn/ontology-{scales_label.replace(' ', '-')}"
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(MODEL_DIR),
                )
                pipe = pipeline(
                    "classification",
                    model=model,
                    tokenizer=model_name,
                    fp16=True,
                )
                data = data.to_dict(orient="records")
                mega_batch_size = 100000
                mega_batches = [
                    data[i : i + mega_batch_size]
                    for i in range(0, len(data), mega_batch_size)
                ]
                for mega_batch in tqdm(
                    mega_batches, desc=f"Predicting {scales_label}"
                ):
                    mega_batch = pd.DataFrame(mega_batch)
                    mega_batch["pred"] = pipe.predict(
                        mega_batch["text"].tolist(),
                        batch_size=16,
                        return_scores=True,
                        max_length=256,
                        truncation=True,
                    )
                    mega_batch["pred"] = mega_batch["pred"].apply(
                        lambda x: x["LABEL_1"]
                    )
                    pd_save_or_append(
                        mega_batch[["text_hash", "pred"]], scales_path
                    )
    else:
        from clx.models import DocketEntry, Label, LabelTag

        project = DocketEntry.get_project()
        for label_name in tqdm(scales2label.values(), desc="Importing tags"):
            label, _ = Label.objects.get_or_create(
                project=project, name=label_name
            )
            tag, _ = LabelTag.objects.get_or_create(label=label, name="scales")
            scales_data = pd.read_csv(
                SCALES_DIR / f"{label2slug(label_name)}.csv"
            )
            text_hashes = list(
                scales_data[scales_data["pred"] > 0.5]["text_hash"].unique()
            )
            label_data = all_data[all_data["text_hash"].isin(text_hashes)]
            print(f"Importing {len(text_hashes)} tags for {label_name}")
            DocketEntry.bulk_replace_tag(tag.id, label_data["id"].tolist())
