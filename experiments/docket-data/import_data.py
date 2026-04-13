import pandas as pd

from clx.models import Project
from clx.settings import CLX_HOME

DATA_PATH = CLX_HOME / "experiments" / "docket-data" / "docket_data.csv"
PREPPED_PATH = (
    CLX_HOME / "experiments" / "docket-data" / "docket_data_prepped.csv"
)

if not PREPPED_PATH.exists():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["meta", "cites"])

    short_data = data[data["short_description"].notna()]
    short_data = short_data.drop(columns=["description"])
    short_data = short_data.rename(columns={"short_description": "text"})
    short_data["is_short_description"] = True
    print(short_data)

    long_data = data[data["description"].notna()]
    long_data = long_data.drop(columns=["short_description"])
    long_data = long_data.rename(columns={"description": "text"})
    long_data["is_short_description"] = False
    print(long_data)

    data = pd.concat([short_data, long_data])
    data = data.drop_duplicates(subset=["text"])
    data = data.dropna(subset=["text"])
    print(data)

    data.to_csv(PREPPED_PATH, index=False)

project, _ = Project.objects.get_or_create(name="Docket Entry")

chunks = pd.read_csv(PREPPED_PATH, chunksize=5000)
for chunk in chunks:
    project.add_docs(chunk)
