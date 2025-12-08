import ast
import re

import pandas as pd
from span_annotator import SpanAnnotationAgent
from tqdm import tqdm

from clx import extract_attachments, pd_save_or_append
from clx.settings import CLX_HOME

PROJECT_DIR = CLX_HOME / "projects" / "entry_citation"
DATA_PATH = PROJECT_DIR / "data.csv"
PREDS_PATH = PROJECT_DIR / "preds.csv"

TASK_DESCRIPTION = """
We are extracting entry number citations from docket entries. For example,
in something like "Order granting 6 motion to dismiss" we would want to
extract the entry number "6".

Sometimes entry numbers will appear in parentheses or brackets. In these cases,
we only want the numeric value. Sometimes the entry number will be hyphenated to
indicate a main entry number / attachment number pair. In this case, you should extract
the full value (including the attachment number).

We only care about extracting entry numbers (sometimes called document numbers)
that reference other docket entries. Do not extract other numbers, like case /
docket numbers, attachment numbers, page numbers, etc. Do not capture numbers that
appear as the very first term in the text, that is not a reference to another docket entry.

Many docket entries will have multiple entry numbers. You should extract all of them.
Be very careful not to miss any. Even if the same entry number appears multiple times,
you should capture all instances.
"""


# Warnings
def invalid_chars(text: str, spans) -> bool:
    for span in spans or []:
        if re.fullmatch(r"\d+(?:-\d+)*", str(span["text"].strip())) is None:
            return True
    return False


def overlapping_spans(text: str, spans) -> bool:
    ordered = sorted(
        [(int(span["start"]), int(span["end"])) for span in spans],
        key=lambda p: (p[0], p[1]),
    )
    last_end = -1
    for start, end in ordered:
        if start < last_end:
            return True
        last_end = max(last_end, end)
    return False


def over_10000(text: str, spans) -> bool:
    for span in spans or []:
        for part in str(span["text"].strip()).split("-"):
            if part.isdigit() and int(part) > 10000:
                return True
    return False


def spans_in_attachments(text: str, spans) -> bool:
    attachment_sections = extract_attachments(text)
    for attachment_section in attachment_sections:
        for span in spans:
            if (
                span["start"] >= attachment_section["start"]
                and span["end"] <= attachment_section["end"]
            ):
                return True
    return False


warnings = {
    "invalid_chars": invalid_chars,
    "overlapping_spans": overlapping_spans,
    "over_10000": over_10000,
    "spans_in_attachments": spans_in_attachments,
}


def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(
            CLX_HOME / "data" / "docket_sample" / "docket_sample.csv"
        )
        data = data[["docket_entry_id", "description"]]
        data = data.rename(
            columns={"docket_entry_id": "id", "description": "text"}
        )
        data = data.dropna(subset=["text"])
        data = data.sample(20000, random_state=42)
        data.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def generate_synthetic_train_data():
    data = load_data()
    if PREDS_PATH.exists():
        existing_ids = pd.read_csv(PREDS_PATH, usecols=["id"])["id"].tolist()
        data = data[~data["id"].isin(existing_ids)]
    data = data.to_dict("records")
    batch_size = 2000
    batches = [
        data[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]
    for batch in tqdm(batches, desc="Predicting"):
        results = SpanAnnotationAgent.apply(
            TASK_DESCRIPTION,
            [row["text"] for row in batch],
            num_workers=128,
            timeout=60,
            num_retries=2,
        )
        batch = pd.DataFrame(batch)
        batch["spans"] = [result["spans"] for result in results]
        batch["status"] = [result["status"] for result in results]
        pd_save_or_append(batch, PREDS_PATH)
    preds = pd.read_csv(PREDS_PATH)
    preds["spans"] = preds["spans"].apply(ast.literal_eval)
    return preds


def load_synthetic_data():
    preds = pd.read_csv(PREDS_PATH)
    preds["spans"] = preds["spans"].apply(ast.literal_eval)
    preds["warnings"] = preds.apply(
        lambda row: [
            w for w, f in warnings.items() if f(row["text"], row["spans"])
        ],
        axis=1,
    )
    return preds


def load_train_eval_data():
    def add_span_label(spans):
        for span in spans:
            span["label"] = "CITATION"
        return spans

    data = load_synthetic_data()
    data = data[
        (data["status"] == "success") & (data["warnings"].apply(len) == 0)
    ]
    data["spans"] = data["spans"].apply(add_span_label)
    data = data.sample(frac=1, random_state=42)
    split = int(0.8 * len(data))
    train_data = data.head(split)
    eval_data = data.tail(-split)
    return train_data, eval_data
