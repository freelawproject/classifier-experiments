import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import runpod
import simplejson as json
import torch

from clx.ml import training_run
from clx.utils import S3


def handler(event):
    """Handle finetuning job requests."""
    start_time = datetime.now()
    inputs = event.pop("input")

    s3_bucket = inputs.pop("s3_bucket")
    s3_prefix = inputs.pop("s3_prefix")
    training_run_args = inputs.pop("training_run")
    overwrite = inputs.pop("overwrite", False)

    s3 = S3(bucket=s3_bucket)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = s3.download_prefix(s3_prefix, tmpdir)
            train_path = Path(data_dir) / "train.csv"
            train_data = pd.read_csv(train_path)
            eval_path = Path(data_dir) / "eval.csv"
            eval_data = pd.read_csv(eval_path) if eval_path.exists() else None

        s3.delete_prefix(s3_prefix)

        run = training_run(**training_run_args)
        run.train(train_data, eval_data, overwrite=overwrite)

        results = {}
        if run.results_path.exists():
            results = json.loads(run.results_path.read_text())

        torch.cuda.empty_cache()

        duration = (datetime.now() - start_time).total_seconds()
        return {
            "status": "success",
            "seconds_elapsed": duration,
            "run_dir": str(run.run_dir),
            "results": results,
        }

    except Exception as e:
        try:
            s3.delete_prefix(s3_prefix)
        except Exception:
            pass

        duration = (datetime.now() - start_time).total_seconds()
        return {
            "status": "error",
            "seconds_elapsed": duration,
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
