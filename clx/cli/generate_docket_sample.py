import ast
import os
import time
import urllib.parse

import click
import pandas as pd
import requests
from tqdm import tqdm

from clx import (
    download_file,
    extract_attachments,
    extract_bz2_file,
    pd_save_or_append,
)
from clx.settings import CLX_HOME

BULK_DOCKETS_URL = os.getenv(
    "BULK_DOCKETS_URL",
    "https://storage.courtlistener.com/bulk-data/dockets-2025-10-31.csv.bz2",
)

PROJECT_DIR = CLX_HOME / "app_projects" / "docket-entry"
BULK_DATA_ZIP_PATH = PROJECT_DIR / "docket_sample" / "recap_dockets.csv.bz2"
BULK_DATA_PATH = PROJECT_DIR / "docket_sample" / "recap_dockets.csv"
COVERAGE_DATA_URL = "https://media.githubusercontent.com/media/freelawproject/classifier-experiments/refs/heads/data/home/data/docket_sample/document_coverage.csv"
COVERAGE_DATA_PATH = PROJECT_DIR / "docket_sample" / "document_coverage.csv"
REDUCED_DATA_PATH = PROJECT_DIR / "docket_sample" / "recap_dockets_reduced.csv"
SAMPLE_INDEX_URL = "https://media.githubusercontent.com/media/freelawproject/classifier-experiments/refs/heads/data/home/data/docket_sample/docket_index.csv"
SAMPLE_INDEX_PATH = PROJECT_DIR / "docket_sample" / "docket_index.csv"
DOCKET_SAMPLE_URL = ""
DOCKET_SAMPLE_PATH = PROJECT_DIR / "docket_sample" / "docket_data.csv"


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


def get_sample_from_reduced_data(
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


def create_docket_index_from_reduced_data():
    """Apply sampling strategies to reduced data to generate the docket index."""
    reduced_data = pd.read_csv(REDUCED_DATA_PATH)
    most_entries_available = reduced_data[
        reduced_data["num_main_available"] / reduced_data["num_main_documents"]
        > 0.9
    ]
    sample = pd.concat(
        [
            # Short cases sample, ~0.5M entries
            get_sample_from_reduced_data(
                reduced_data, 10, ["filing_year", "court_id"], max_entries=20
            ),
            # Medium cases sample, ~7M entries
            get_sample_from_reduced_data(
                reduced_data,
                20,
                ["filing_year", "court_id"],
                min_entries=20,
                max_entries=500,
            ),
            # Long cases sample, filing year group only, ~1M entries
            get_sample_from_reduced_data(
                reduced_data,
                20,
                ["filing_year"],
                min_entries=500,
                max_entries=5000,
            ),
            # Cases where at least 90% of the main documents are available, ~0.5M entries
            get_sample_from_reduced_data(
                most_entries_available,
                100,
                ["filing_year", "court_id"],
                min_entries=10,
                max_entries=5000,
            ),
            # Crude Case Type sample, ~1M entries
            get_sample_from_reduced_data(
                reduced_data, 5000, ["crude_case_type"], max_entries=1000
            ),
            # Nature of Suit sample, ~1.5M entries
            get_sample_from_reduced_data(
                reduced_data, 30, ["nature_of_suit"], max_entries=1000
            ),
        ]
    )
    sample = sample.drop_duplicates(subset=["id"])
    sample = sample[["id", "filing_year", "num_documents"]]
    sample.to_csv(SAMPLE_INDEX_PATH, index=False)
    print(f"Saved docket index to {SAMPLE_INDEX_PATH}")
    print(f"Number of unique dockets: {sample['id'].nunique()}")
    print(f"Number of docket entries: {sample['num_documents'].sum()}")


def create_docket_index_from_scratch():
    """The full flow to create the docket index from scratch."""
    if not REDUCED_DATA_PATH.exists():
        if not COVERAGE_DATA_PATH.exists():
            # Download coverage data
            print("Downloading coverage data...")
            download_file(COVERAGE_DATA_URL, COVERAGE_DATA_PATH)
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
    create_docket_index_from_reduced_data()


def get_with_retry(url, headers, max_retries=3, sleep=10, rate_limit_sleep=90):
    """Get a URL with retry and sleep."""
    for _ in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                print(
                    f"Rate limit exceeded, sleeping for {rate_limit_sleep} seconds..."
                )
                time.sleep(rate_limit_sleep)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting {url}: {e}")
            print(f"Sleeping for {sleep} seconds...")
            time.sleep(sleep)
    raise Exception(f"Too many retries for {url}")


def download_single_docket_data(docket_id, progress):
    """Download a docket from CourtListener."""
    time.sleep(0.3)
    cl_token = os.getenv("CL_TOKEN")
    assert cl_token is not None, "CL_TOKEN is not set"
    headers = {"Authorization": f"Token {cl_token}"}

    base_url = "https://www.courtlistener.com/api/rest/v4/search/"
    query = {
        "q": f"docket_id:{docket_id}",
        "type": "rd",
        "order_by": "entry_date_filed asc",
    }
    url = base_url + "?" + urllib.parse.urlencode(query, safe=":")

    page_data = get_with_retry(url, headers)
    progress.update(len(page_data["results"]))
    data = page_data["results"]
    while page_data["next"]:
        time.sleep(0.3)
        page_data = get_with_retry(page_data["next"], headers)
        progress.update(len(page_data["results"]))
        data.extend(page_data["results"])
    data = pd.DataFrame(data)
    data = data[
        [
            "absolute_url",
            "attachment_number",
            "cites",
            "description",
            "docket_entry_id",
            "docket_id",
            "document_number",
            "document_type",
            "entry_date_filed",
            "entry_number",
            "filepath_local",
            "id",
            "is_available",
            "meta",
            "pacer_doc_id",
            "page_count",
            "short_description",
            "snippet",
        ]
    ]
    return data


def download_docket_sample():
    docket_index = pd.read_csv(SAMPLE_INDEX_PATH)
    docket_index = docket_index.sort_values(
        by="num_documents", ascending=False
    )

    total_entries = docket_index["num_documents"].sum()

    if DOCKET_SAMPLE_PATH.exists():
        existing_ids = pd.read_csv(DOCKET_SAMPLE_PATH, usecols=["docket_id"])
        existing_ids = existing_ids.drop_duplicates()
        docket_index = docket_index[
            ~docket_index["id"].isin(existing_ids["docket_id"])
        ]

    if len(docket_index) > 0:
        remaining_entries = docket_index["num_documents"].sum()
        print(f"{total_entries - remaining_entries} entries downloaded so far")
        progress = tqdm(
            desc="Downloading docket sample",
            total=remaining_entries,
            unit=" entries",
        )
        for row in docket_index.to_dict("records"):
            docket_data = download_single_docket_data(row["id"], progress)
            pd_save_or_append(docket_data, DOCKET_SAMPLE_PATH)
        progress.close()
    print(f"All docket sample data downloaded to {DOCKET_SAMPLE_PATH}")


def parse_attachments(text):
    """Parse the attachments from a docket entry."""
    attachment_sections = extract_attachments(text)
    attachment_sections = sorted(
        attachment_sections, key=lambda x: -x["start"]
    )
    attachments = []
    for attachment_section in attachment_sections:
        text = (
            text[: attachment_section["start"]]
            + " "
            + text[attachment_section["end"] :]
        )
        text = " ".join(text.split()).strip()
        attachments.extend(attachment_section["attachments"])
    return text, attachments


def import_docket_sample(batch_size=200000):
    from clx.models import DocketEntry, DocketEntryShort

    total_entries = pd.read_csv(SAMPLE_INDEX_PATH)["num_documents"].sum()

    shorts = pd.DataFrame(columns=["text", "text_type", "count"])
    progress = tqdm(
        desc="Importing docket sample", total=total_entries, unit=" entries"
    )
    for data in pd.read_csv(
        DOCKET_SAMPLE_PATH, chunksize=batch_size, low_memory=False
    ):
        progress.update(len(data))
        data["timestamp"] = pd.to_datetime(
            data["meta"].apply(lambda x: ast.literal_eval(x)["timestamp"]),
            format="mixed",
        )
        # Filter to only PACER documents, we will extract attachments with regex
        data = data[data["document_type"] == "PACER Document"]
        # Get the short descriptions before dropping where description is null
        short_data = data[["short_description"]].dropna()
        if len(short_data) > 0:
            short_data = short_data.rename(
                columns={"short_description": "text"}
            )
            short_data = short_data["text"].value_counts().reset_index()
            short_data.columns = ["text", "update_count"]
            short_data["text_type"] = "short_description"
            shorts = shorts.merge(
                short_data, on=["text", "text_type"], how="outer"
            )
            shorts["count"] = shorts["count"].fillna(0) + shorts[
                "update_count"
            ].fillna(0)
            shorts = shorts.drop(columns=["update_count"])
        data = data.dropna(subset="description")
        if len(data) > 0:
            # Rare, but sometimes there are duplicates for main documents, keep the most recent
            data = data.sort_values(by="timestamp", ascending=False)
            data = data.drop_duplicates(
                subset=["docket_entry_id"], keep="first"
            )
            # Simplify main document text and extract attachment text
            data["text"], data["attachments"] = zip(
                *data["description"].apply(parse_attachments)
            )
            # Add attachments to shorts
            attachments = data[["attachments"]].explode("attachments").dropna()
            if len(attachments) > 0:
                attachments["text"] = attachments["attachments"].apply(
                    lambda x: x["attachment_description"]
                )
                attachments = attachments["text"].value_counts().reset_index()
                attachments.columns = ["text", "update_count"]
                attachments["text_type"] = "attachment"
                shorts = shorts.merge(
                    attachments, on=["text", "text_type"], how="outer"
                )
                shorts["count"] = shorts["count"].fillna(0) + shorts[
                    "update_count"
                ].fillna(0)
                shorts = shorts.drop(columns=["update_count"])
            # Rename and filter columns before pushing to the database
            data = data.rename(
                columns={
                    "id": "recap_id",
                    "docket_entry_id": "id",
                    "entry_date_filed": "date_filed",
                }
            )
            data = data[
                [
                    "id",
                    "recap_id",
                    "docket_id",
                    "entry_number",
                    "date_filed",
                    "text",
                ]
            ]
            data["date_filed"] = pd.to_datetime(data["date_filed"])
            data["entry_number"] = data["entry_number"].astype(pd.Int64Dtype())
            data = data[data["text"].apply(len) > 0]
            # Push to the database
            if len(data) > 0:
                DocketEntry.bulk_insert(data, ignore_conflicts=True)
    # Push shorts to the database and update counts
    shorts["count"] = shorts["count"].fillna(0).astype(int)
    DocketEntryShort.bulk_insert(shorts, ignore_conflicts=True)
    DocketEntryShort.bulk_update_column(
        "count",
        shorts["text"].tolist(),
        shorts["count"].tolist(),
        id_column="text",
    )


@click.command()
@click.option(
    "do_import", "--import", is_flag=True, help="Import the docket sample"
)
@click.option(
    "--skip-cached-steps",
    is_flag=True,
    help="Instead of downloading the cached data, this will re-run the data generation pipeline from scratch.",
)
def generate_docket_sample(do_import, skip_cached_steps):
    """Prepare a sample of dockets for use in the application."""
    DOCKET_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Get the case index
    if not SAMPLE_INDEX_PATH.exists():
        if not skip_cached_steps:
            print("Downloading cached sample index...")
            download_file(SAMPLE_INDEX_URL, SAMPLE_INDEX_PATH)
        else:
            create_docket_index_from_scratch()

    # Get the sample data
    if not DOCKET_SAMPLE_PATH.exists() and not skip_cached_steps:
        print("Downloading cached docket sample data...")
        download_file(DOCKET_SAMPLE_URL, DOCKET_SAMPLE_PATH)

    # Check for gaps and download from CourtListener
    print("Downloading docket sample from CourtListener...")

    # Temporary, later remove the blocker to download_docket_sample()
    if not do_import:
        download_docket_sample()
    else:
        import_docket_sample()
