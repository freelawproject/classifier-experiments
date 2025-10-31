import os

import click

from clx import download_file, extract_bz2_file
from clx.settings import LOCAL_DATA_DIR

BULK_DOCKETS_URL = os.getenv(
    "BULK_DOCKETS_URL",
    "https://storage.courtlistener.com/bulk-data/dockets-2025-10-28.csv.bz2",
)


@click.command()
def generate_docket_sample():
    """Prepare a sample of dockets for use in the application."""
    bulk_data_zip_path = LOCAL_DATA_DIR / "sample" / "recap_dockets.csv.bz2"
    bulk_data_path = LOCAL_DATA_DIR / "sample" / "recap_dockets.csv"
    if not bulk_data_path.exists():
        if not bulk_data_zip_path.exists():
            print(
                f"Downloading bulk dockets data to {bulk_data_zip_path}\nThis may take a while..."
            )
            download_file(BULK_DOCKETS_URL, bulk_data_zip_path)

        print(
            f"Extracting {bulk_data_zip_path} to {bulk_data_path}\nThis may take a while..."
        )
        extract_bz2_file(bulk_data_zip_path, bulk_data_path)
