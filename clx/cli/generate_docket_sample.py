import os

import click
import pandas as pd
from tqdm import tqdm

from clx import download_file, extract_bz2_file, pd_save_or_append
from clx.settings import LOCAL_DATA_DIR

BULK_DOCKETS_URL = os.getenv(
    "BULK_DOCKETS_URL",
    "https://storage.courtlistener.com/bulk-data/dockets-2025-10-28.csv.bz2",
)
BULK_DATA_ZIP_PATH = LOCAL_DATA_DIR / "docket_sample" / "recap_dockets.csv.bz2"
BULK_DATA_PATH = LOCAL_DATA_DIR / "docket_sample" / "recap_dockets.csv"
COVERAGE_DATA_PATH = LOCAL_DATA_DIR / "docket_sample" / "document_coverage.csv"
REDUCED_DATA_PATH = (
    LOCAL_DATA_DIR / "docket_sample" / "recap_dockets_reduced.csv"
)
SAMPLE_INDEX_PATH = LOCAL_DATA_DIR / "docket_sample" / "docket_index.csv"


def get_crude_case_type(docket_number):
    """Find common two letter identifiers as a crude proxy for case type."""
    case_types = [
        "cv",
        "bk",
        "cr",
        "mj",
        "po",
        "mc",
        "ap",
        "sw",
        "vv",
        "dp",
        "pq",
        "gj",
        "mb",
    ]
    parts = [x.lower() for x in docket_number.split("-")]
    for case_type in case_types:
        if case_type in parts:
            return case_type
    return "other"


def reduce_bulk_data():
    """Reduce and preprocess bulk dockets data."""
    selected_cols = [
        "id",
        "docket_number",
        "date_filed",
        "court_id",
        "nature_of_suit",
        "jurisdiction_type",
    ]

    coverage_data = pd.read_csv(COVERAGE_DATA_PATH)
    stream = pd.read_csv(
        BULK_DATA_PATH, chunksize=300000, usecols=selected_cols
    )

    progress = tqdm(desc="Collected", unit=" dockets")
    for chunk in stream:
        chunk = chunk.merge(coverage_data, on="id", how="inner")
        chunk["filing_year"] = pd.to_datetime(chunk["date_filed"]).dt.year
        chunk["crude_case_type"] = (
            chunk["docket_number"].fillna("").apply(get_crude_case_type)
        )
        pd_save_or_append(chunk, REDUCED_DATA_PATH)
        progress.update(len(chunk))


def get_sample(
    reduced_data, n, group_cols, min_entries=None, max_entries=None
):
    """Get a sample from the reduced data based on some grouping columns."""
    sample = reduced_data.copy()
    if min_entries is not None:
        sample = sample[sample["num_main_documents"] > min_entries]
    if max_entries is not None:
        sample = sample[sample["num_main_documents"] < max_entries]
    sample = sample.groupby(group_cols)
    sample = sample.apply(lambda x: x.sample(min(len(x), n)))
    return sample.reset_index(drop=True)


def create_docket_index():
    """Apply sampling strategies to generate the docket index."""
    reduced_data = pd.read_csv(REDUCED_DATA_PATH)
    most_entries_available = reduced_data[
        reduced_data["num_main_available"] / reduced_data["num_main_documents"]
        > 0.9
    ]
    sample = pd.concat(
        [
            # Short cases sample, ~0.5M entries
            get_sample(
                reduced_data, 10, ["filing_year", "court_id"], max_entries=20
            ),
            # Medium cases sample, ~7M entries
            get_sample(
                reduced_data,
                20,
                ["filing_year", "court_id"],
                min_entries=20,
                max_entries=500,
            ),
            # Long cases sample, filing year group only, ~1M entries
            get_sample(
                reduced_data,
                20,
                ["filing_year"],
                min_entries=500,
                max_entries=5000,
            ),
            # Cases where at least 90% of the main documents are available, ~0.5M entries
            get_sample(
                most_entries_available,
                100,
                ["filing_year", "court_id"],
                min_entries=10,
                max_entries=5000,
            ),
            # Crude Case Type sample, ~1M entries
            get_sample(
                reduced_data, 5000, ["crude_case_type"], max_entries=1000
            ),
            # Nature of Suit sample, ~1.5M entries
            get_sample(reduced_data, 30, ["nature_of_suit"], max_entries=1000),
        ]
    )
    sample = sample.drop_duplicates(subset=["id"])
    sample = sample[["id", "filing_year", "num_documents"]]
    sample.to_csv(SAMPLE_INDEX_PATH, index=False)
    print(f"Saved docket index to {SAMPLE_INDEX_PATH}")
    print(f"Number of unique dockets: {sample['id'].nunique()}")
    print(f"Number of docket entries: {sample['num_documents'].sum()}")


@click.command()
def generate_docket_sample():
    """Prepare a sample of dockets for use in the application."""
    if not SAMPLE_INDEX_PATH.exists():
        if not REDUCED_DATA_PATH.exists():
            if not BULK_DATA_PATH.exists():
                if not BULK_DATA_ZIP_PATH.exists():
                    # Download the bulk dockets data
                    print(
                        f"Downloading bulk dockets data to {BULK_DATA_ZIP_PATH}\nThis may take a while..."
                    )
                    download_file(BULK_DOCKETS_URL, BULK_DATA_ZIP_PATH)

                # Extract the bulk dockets data
                print(
                    f"Extracting {BULK_DATA_ZIP_PATH} to {BULK_DATA_PATH}\nThis may take a while..."
                )
                extract_bz2_file(BULK_DATA_ZIP_PATH, BULK_DATA_PATH)

            # Reduce the bulk dockets data
            print("Reduce and preprocess bulk dockets data...")
            reduce_bulk_data()

        # Create the docket index
        print("Creating docket index...")
        create_docket_index()
