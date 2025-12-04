import gc
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from clx import pd_save_or_append
from clx.settings import CLX_HOME, DATA_DIR

PROJECT_DIR = CLX_HOME / "projects" / "docketbert"


def tokenize(examples):
    return tokenizer(
        examples["text"], padding=False, truncation=True, max_length=768
    )


for use_cpu in [True, False]:
    out_path = PROJECT_DIR / "inference_test.csv"
    if use_cpu:
        out_path = out_path.with_suffix(".cpu.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = list((PROJECT_DIR / "runs").glob("docketbert-*/model"))
    paths = paths + ["microsoft/deberta-v3-large"]
    texts = (
        pd.read_csv(
            DATA_DIR / "search_datasets" / "docket-entry.csv", nrows=100000
        )["text"]
        .dropna()
        .tolist()
    )
    if use_cpu:
        texts = texts[:200]
    completed = {}
    if out_path.exists():
        data = pd.read_csv(out_path).to_dict(orient="records")
        for row in data:
            completed[row["model"]] = completed.get(row["model"], []) + [
                row["batch_size"]
            ]

    for path in paths:
        model_name = str(path)
        if isinstance(path, Path):
            model_name = path.parent.name
        if use_cpu:
            batch_sizes = [32]
        else:
            batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        for batch_size in batch_sizes:
            if batch_size not in completed.get(model_name, []):
                model = AutoModel.from_pretrained(
                    path,
                    attn_implementation="eager" if use_cpu else None,
                ).to("cpu" if use_cpu else "cuda")
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(path)
                dataset = Dataset.from_dict({"text": texts})
                dataset = dataset.map(tokenize, batched=True, num_proc=8)
                dataset = dataset.select_columns(
                    ["input_ids", "attention_mask"]
                )
                collator = DataCollatorWithPadding(
                    tokenizer, pad_to_multiple_of=8
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collator,
                )

                try:
                    autocast_context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if not use_cpu
                        else nullcontext()
                    )
                    with autocast_context, torch.inference_mode():
                        if not use_cpu:
                            torch.cuda.synchronize()
                        start_time = time.time()
                        for batch in tqdm(
                            loader, desc=f"{model_name} ({batch_size})"
                        ):
                            inputs = {
                                k: v.to(model.device) for k, v in batch.items()
                            }
                            outputs = model(**inputs)
                        if not use_cpu:
                            torch.cuda.synchronize()
                        duration = (
                            str(int((time.time() - start_time) * 1000) / 1000)
                            + "s"
                        )
                except torch.OutOfMemoryError:
                    duration = "OOM"
                del model
                gc.collect()
                if not use_cpu:
                    torch.cuda.empty_cache()
                data = pd.DataFrame(
                    [
                        {
                            "model": model_name,
                            "batch_size": batch_size,
                            "duration": duration,
                        }
                    ]
                )
                pd_save_or_append(data, out_path)
                print(f"{model_name} ({batch_size}): {duration}")
