import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd

from clx.ml import training_run


class TrainingRunTest(TestCase):
    """Quick training run tests on dummy data."""

    def test_classification_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = []
            for label in ["apple", "banana", "orange"]:
                data += [{"text": label, "label": "fruit"} for _ in range(300)]
            for label in ["dog", "cat", "bird"]:
                data += [
                    {"text": label, "label": "not fruit"} for _ in range(300)
                ]

            data = pd.DataFrame(data).sample(frac=1)

            split = int(0.8 * len(data))
            train_data = data.head(split)
            eval_data = data.tail(-split)
            label_names = sorted(data["label"].unique())

            run = training_run(
                "classification",
                run_name="test",
                run_dir_parent=temp_dir,
                base_model_name="docketanalyzer/modernbert-unit-test",
                label_names=label_names,
                training_args={
                    "eval_strategy": "steps",
                    "eval_steps": 100,
                    "save_steps": 100,
                },
            )

            run.train(train_data, eval_data, overwrite=True)

            run = training_run(load=Path(temp_dir) / "test")

            eval_data["preds"] = run.predict(
                eval_data["text"].tolist(), batch_size=8
            )
            eval_data["preds"] = eval_data["preds"].apply(
                lambda x: max(x, key=x.get)
            )

            acc = (eval_data["preds"] == eval_data["label"]).mean()
            self.assertEqual(acc, 1.0)
