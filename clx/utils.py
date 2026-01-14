import bz2
import hashlib
import os
from pathlib import Path

import boto3
import django
import pandas as pd
import regex as re
import requests
import simplejson as json
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


class S3:
    """S3 client wrapper for upload/download/delete operations."""

    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or os.getenv("CLX_S3_BUCKET")
        if not self.bucket:
            raise ValueError(
                "S3 bucket must be provided or set via CLX_S3_BUCKET env var"
            )
        self._client = None

    @property
    def client(self):
        """Lazy-loaded boto3 S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=os.getenv("CLX_S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("CLX_S3_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("CLX_S3_SECRET_ACCESS_KEY"),
                region_name=os.getenv("CLX_S3_REGION"),
            )
        return self._client

    def ping(self) -> bool:
        """Test connection to S3 bucket. Returns True if successful."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    def upload(self, local_path: Path | str, key: str) -> str:
        """Upload file to S3. Returns S3 URI."""
        self.client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download(self, key: str, local_path: Path | str) -> Path:
        """Download file from S3 to local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))
        return local_path

    def download_prefix(self, prefix: str, local_dir: Path | str) -> Path:
        """Download all objects under an S3 prefix to a local directory."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_path = key[len(prefix) :].lstrip("/")
                if relative_path:
                    local_path = local_dir / relative_path
                    self.download(key, local_path)
        return local_dir

    def delete(self, key: str) -> None:
        """Delete single object from S3."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        """Delete all objects under an S3 prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": objects}
                )
