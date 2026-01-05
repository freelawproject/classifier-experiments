from datetime import datetime

import runpod
import torch

from clx import generate_hash
from clx.ml import pipeline

PIPELINE = {"id": None, "pipe": None}


def load_pipeline(pipeline_args):
    pipe_id = generate_hash(pipeline_args)
    if pipe_id != PIPELINE["id"]:
        del PIPELINE["pipe"]
        torch.cuda.empty_cache()
        PIPELINE["pipe"] = pipeline(**pipeline_args)
    return PIPELINE["pipe"]


def handler(event):
    start_time = datetime.now()
    inputs = event.pop("input")
    pipe = load_pipeline(inputs.pop("pipeline"))
    texts = inputs.pop("texts")
    try:
        results = pipe(texts, **inputs)
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "results": results,
            "seconds_elapsed": duration,
            "num_examples": len(texts),
            "status": "success",
        }
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "results": None,
            "seconds_elapsed": duration,
            "num_examples": len(texts),
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
