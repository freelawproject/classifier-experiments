import os
from pathlib import Path

import pandas as pd
import psycopg2
from tqdm import tqdm

from clx import pd_save_or_append
from clx.settings import CLX_HOME

PROJECT_DIR = CLX_HOME / "projects" / "docketbert"

DB_CONFIG = {
    "host": os.getenv("DEV_DB_HOST"),
    "port": int(os.getenv("DEV_DB_PORT", "5432")),
    "dbname": os.getenv("DEV_DB_NAME"),
    "user": os.getenv("DEV_DB_USER"),
    "password": os.getenv("DEV_DB_PASSWORD"),
}


def pull_dev_data(table_name, nrows=5_000_000, batch_size=1_000_000) -> None:
    data_path = Path(PROJECT_DIR / "data" / f"{table_name}_descriptions.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**DB_CONFIG)

    last_id = None
    current_rows = 0
    if data_path.exists():
        chunks = pd.read_csv(data_path, chunksize=batch_size)
        for chunk in chunks:
            current_rows += len(chunk)
            min_id = chunk["id"].min()
            last_id = min_id if last_id is None else min(last_id, min_id)

    progress = tqdm(total=nrows, desc="Downloading")
    progress.update(current_rows)

    try:
        while current_rows < nrows:
            last_id_condition = (
                "" if last_id is None else f"AND id < {last_id}"
            )
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT id, description FROM {table_name}
                WHERE description IS NOT NULL
                AND description <> ''
                {last_id_condition}
                ORDER BY id DESC
                LIMIT {batch_size}
                """)
                rows = cur.fetchall()
                if not rows:
                    break
                col_names = [desc[0] for desc in cur.description]
                data = pd.DataFrame(rows, columns=col_names)
                last_id = data["id"].min()
                current_rows += len(data)
                progress.update(len(data))
                pd_save_or_append(data, data_path)
    finally:
        conn.close()


def consolidate_data():
    d1 = pd.read_csv(CLX_HOME / "app_projects" / "docket-entry" / "docs.csv")[
        "text"
    ]
    d2 = pd.read_csv(
        CLX_HOME / "app_projects" / "docket-entry-short" / "docs.csv"
    )["text"]
    d3 = pd.read_csv(
        PROJECT_DIR / "data" / "search_recapdocument_descriptions.csv",
        usecols=["description"],
    )
    d3 = d3.rename(columns={"description": "text"})
    d4 = pd.read_csv(
        PROJECT_DIR / "data" / "search_docketentry_descriptions.csv",
        usecols=["description"],
    )
    d4 = d4.rename(columns={"description": "text"})
    data = pd.concat([d1, d2, d3, d4])
    data = data.drop_duplicates("text")
    data = data.sample(frac=1)
    return data


if __name__ == "__main__":
    pull_dev_data("search_docketentry", nrows=40_000_000)
    pull_dev_data("search_recapdocument", nrows=20_000_000)
    data = consolidate_data()
    data.to_csv(PROJECT_DIR / "data" / "train.csv", index=False)
    print(data)
