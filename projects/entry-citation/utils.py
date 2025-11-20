import pandas as pd

from clx.settings import CLX_HOME

DATA_PATH = CLX_HOME / "projects" / "entry_citation" / "data.csv"


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
