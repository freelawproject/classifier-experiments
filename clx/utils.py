import bz2
import hashlib
import json
import os
from pathlib import Path

import django
import pandas as pd
import regex as re
import requests
from tqdm import tqdm

tqdm.pandas()


def init_django():
    """Initializes Django."""
    os.environ["DJANGO_SETTINGS_MODULE"] = "clx.settings"
    django.setup()


def generate_hash(data):
    """Generate a hash of the data."""
    return hashlib.sha256(
        json.dumps({"data": data}, sort_keys=True).encode()
    ).hexdigest()


def label2slug(label_name):
    """Convert a label name to a slug."""
    return label_name.lower().replace(" ", "_").replace("/", "-")


def pd_save_or_append(data: pd.DataFrame, path: str | Path):
    """Save or append pandas dataframe to csv file."""
    if path.exists():
        data.to_csv(path, index=False, mode="a", header=False)
    else:
        data.to_csv(path, index=False)


def download_file(
    url: str, path: str | Path, description: str = "Downloading"
):
    """Download file from URL to local path with progress bar."""
    path = Path(path)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        path.open("wb") as file,
        tqdm(
            desc=description,
            total=total_size,
            unit="iB",
            unit_scale=True,
        ) as progress,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress.update(size)


def extract_bz2_file(
    zip_path: str | Path, path: str | Path, description: str = "Extracting"
):
    """Unzip file from zip path to path with progress bar."""
    path = Path(path)

    with (
        bz2.BZ2File(zip_path, "rb") as zip_file,
        path.open("wb") as file,
        tqdm(desc=description, unit="iB", unit_scale=True) as progress,
    ):
        for data in iter(lambda: zip_file.read(1024 * 1024), b""):
            file.write(data)
            progress.update(len(data))


def extract_from_pattern(
    text, pattern, label, ignore_case=False, extract_groups=None
):
    """Extract spans from text using a regex pattern."""
    spans = []
    for match in re.finditer(
        pattern, text, re.IGNORECASE if ignore_case else 0
    ):
        spans.append(
            {
                "start": match.start(),
                "end": match.end(),
                "label": label,
            }
        )
        if extract_groups:
            for k, v in extract_groups.items():
                spans[-1][k] = {
                    "start": match.start(v),
                    "end": match.end(v),
                }
    return spans


def extract_attachments(text):
    """Parse the attachment sections from docket entries."""
    pattern = (
        r"((\(|\( )?(EXAMPLE: )?(additional )?Attachment\(?s?\)?"
        r"([^:]+)?: )((([^()]+)?(\(([^()]+|(?7))*+\))?([^()]+)?)*+)\)*+"
    )
    spans = extract_from_pattern(
        text,
        pattern,
        "attachment_section",
        ignore_case=True,
        extract_groups={"attachments": 6},
    )
    for span in spans:
        attachments = []
        attachments_start = span["attachments"]["start"]
        attachments_end = span["attachments"]["end"]
        attachments_str = text[attachments_start:attachments_end]
        for attachment in re.finditer(
            r"# (\d+) ([^#]+?)(?=, #|#|$)", attachments_str
        ):
            attachments.append(
                {
                    "attachment_number": attachment.group(1),
                    "attachment_description": attachment.group(2),
                    "start": attachments_start + attachment.start(),
                    "end": attachments_start + attachment.end(),
                    "label": "attachment",
                }
            )
        span["attachments"] = attachments
    return spans
