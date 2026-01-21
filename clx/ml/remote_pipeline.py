import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


class RemotePipeline:
    def __init__(
        self,
        endpoint_id: str | None = None,
        api_key: str | None = None,
        **pipeline_args: dict,
    ):
        self.pipeline_args = pipeline_args
        self.endpoint_id = endpoint_id or os.getenv(
            "RUNPOD_INFERENCE_ENDPOINT_ID"
        )
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.endpoint_id or not self.api_key:
            raise Exception(
                "RUNPOD_INFERENCE_ENDPOINT_ID and RUNPOD_API_KEY must be set"
            )

    def predict(
        self,
        examples: list,
        batch_size: int = 16,
        megabatch_size: int = 1024,
        num_workers: int = 8,
        num_retries=3,
        sleep=5,
        **kwargs: dict,
    ):
        def handle_megabatch(megabatch: list):
            errors = []
            for _ in range(num_retries):
                payload = {
                    "input": {
                        "pipeline": self.pipeline_args,
                        "examples": megabatch,
                        "batch_size": batch_size,
                        **kwargs,
                    }
                }
                try:
                    response = requests.post(
                        f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    outputs = response.json()
                except Exception as e:
                    outputs = {"status": "FAILED", "error": str(e)}
                if outputs["status"] != "COMPLETED":
                    errors.append(outputs)
                else:
                    return {
                        "results": outputs["output"]["results"],
                        "status": "success",
                    }
                time.sleep(sleep)
            errors.append(f"Failed to predict after {num_retries} retries")
            return {"results": None, "status": "error", "errors": errors}

        megabatches = [
            examples[i : i + megabatch_size]
            for i in range(0, len(examples), megabatch_size)
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(handle_megabatch, megabatch)
                for megabatch in megabatches
            ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Predicting",
            disable=len(futures) < 2,
        ):
            result = future.result()
            if result["status"] == "error":
                raise Exception(result["errors"])

        results = []
        for future in futures:
            results += future.result()["results"]
        return results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
