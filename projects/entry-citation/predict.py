import torch
from utils import PROJECT_DIR, load_train_eval_data

from clx.ml import pipeline

if __name__ == "__main__":
    _, eval_data = load_train_eval_data()

    model_dir = PROJECT_DIR / "runs" / "entry-citation" / "model"

    pipe = pipeline("ner", model=model_dir)
    with torch.autocast(pipe.pipe.device.type, dtype=torch.bfloat16):
        eval_data["preds"] = pipe.predict(
            eval_data["text"].tolist(),
            batch_size=32,
            max_length=768,
            truncation=True,
        )

    for row in eval_data.to_dict("records"):
        print(row["text"], "\n")
        for span in row["preds"]:
            print(span["text"], "|", float(span["score"]))
        print("-" * 60)
