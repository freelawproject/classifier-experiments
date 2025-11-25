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
                eval_data["text"].tolist(),
                batch_size=8,
            )

            acc = (eval_data["preds"] == eval_data["label"]).mean()
            self.assertEqual(acc, 1.0)

    def test_multi_label_classification_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_labels = {
                "apple": ["fruit", "red"],
                "banana": ["fruit", "yellow"],
                "sun": ["yellow"],
            }

            data = []
            for text, labels in text_labels.items():
                data += [{"text": text, "labels": labels} for _ in range(500)]

            data = pd.DataFrame(data).sample(frac=1)

            split = int(0.8 * len(data))
            train_data = data.head(split)
            eval_data = data.tail(-split)
            label_names = sorted(data["labels"].explode().dropna().unique())

            run = training_run(
                "multi-label-classification",
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

            eval_data["preds"] = run.predict(
                eval_data["text"].tolist(),
                batch_size=8,
            )
            eval_data["labels"] = eval_data["labels"].apply(sorted)
            eval_data["preds"] = eval_data["preds"].apply(sorted)

            acc = (eval_data["preds"] == eval_data["labels"]).mean()
            self.assertEqual(acc, 1.0)

    def test_token_classification_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            texts = [
                "John Doe is a person.",
                "She gave John Doe a taco.",
                "John Doe is happy.",
                "Hello, John Doe",
            ]

            data = []

            for text in texts:
                start = text.index("John Doe")
                for _ in range(300):
                    data.append(
                        {
                            "text": text,
                            "spans": [
                                {
                                    "start": start,
                                    "end": start + 8,
                                    "label": "name",
                                }
                            ],
                        }
                    )

            data = pd.DataFrame(data).sample(frac=1)

            split = int(0.8 * len(data))
            train_data = data.head(split)
            eval_data = data.tail(-split)
            label_names = ["name"]

            run = training_run(
                "ner",
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
            # Remove once pipelines are implemented
            return

            eval_data["preds"] = run.predict(
                eval_data["text"].tolist(),
                batch_size=8,
            )
            eval_data["labels"] = eval_data["labels"].apply(sorted)
            eval_data["preds"] = eval_data["preds"].apply(sorted)

            acc = (eval_data["preds"] == eval_data["labels"]).mean()
            self.assertEqual(acc, 1.0)
