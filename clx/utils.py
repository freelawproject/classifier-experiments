import bz2
import os
from pathlib import Path

import django
import requests
from tqdm import tqdm


def init_django():
    os.environ["DJANGO_SETTINGS_MODULE"] = "clx.settings"
    django.setup()


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
