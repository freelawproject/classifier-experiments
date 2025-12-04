import json

import pandas as pd

from clx.settings import BASE_DIR, CLX_HOME

PROJECT_DIR = CLX_HOME / "projects" / "docketbert"


def prep_inference_data(path):
    inference_data = pd.read_csv(path)
    inference_data["duration"] = inference_data["duration"].apply(
        lambda x: float(x.replace("s", "")) if x != "OOM" else None
    )
    inference_data = (
        inference_data.groupby("model")["duration"].min().reset_index()
    )
    inference_data = inference_data.sort_values("duration")
    return inference_data


def load_eval_loss(model_name):
    path = PROJECT_DIR / "runs" / model_name / "checkpoints" / "results.json"
    if path.exists():
        results = json.loads(path.read_text())
        return results["eval_loss"]


gpu_data = prep_inference_data(PROJECT_DIR / "inference_test.csv")
gpu_data = gpu_data.rename(columns={"duration": "100k_gpu_seconds"})
cpu_data = prep_inference_data(PROJECT_DIR / "inference_test.cpu.csv")
cpu_data = cpu_data.rename(columns={"duration": "200_cpu_seconds"})
data = pd.merge(gpu_data, cpu_data, on="model")
data["num_params_million"] = data["model"].apply(
    lambda x: None if "M" not in x else int(x.split("-")[-1][:-1])
)
data["eval_loss"] = data["model"].apply(load_eval_loss)
data = data.sort_values("eval_loss")
readme_path = BASE_DIR.parent / "projects" / "docketbert" / "README.md"
readme_template_path = readme_path.with_suffix(".template.md")
template = readme_template_path.read_text()
readme = template.format(results_table=data.to_markdown())
readme_path.write_text(readme)
