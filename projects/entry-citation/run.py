from span_annotator import SpanAnnotationAgent
from utils import load_data

task_description = """
We are extracting entry number citations from docket entries. For example,
in something like "Order granting 6 motion to dismiss" we would want to
extract the entry number "6".

Sometimes entry numbers will appear in parentheses or brackets. In these cases,
we only want the numeric value. Sometimes the entry number will be hyphenated to
indicate a main entry number / attachment number pair. In this case, you should extract
the full value (including the attachment number).

We only care about extracting entry numbers (sometimes called document numbers)
that reference other docket entries. Do not extract other numbers, like case /
docket numbers, attachment numbers, page numbers, etc.

Many docket entries will have multiple entry numbers. You should extract all of them.
Be very careful not to miss any.
"""

data = load_data().sample(100)
results = SpanAnnotationAgent.apply(
    task_description,
    data["text"].tolist(),
    num_workers=32,
)
data["spans"] = [result["spans"] for result in results]
data["status"] = [result["status"] for result in results]

for row in data.to_dict("records"):
    print(row["text"])
    for span in row["spans"]:
        print(span)
    print(row["status"])
    print("-" * 100, "\n\n")
