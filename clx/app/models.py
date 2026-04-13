import json
import random
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
from django.conf import settings as django_settings
from django.contrib.postgres.indexes import GinIndex
from django.db import models
from django.utils import timezone
from django_shortuuid.fields import ShortUUIDField
from shortuuid import uuid
from tqdm import tqdm

from clx.settings import CLX_HOME
from clx.utils import generate_hash, pd_save_or_append

from .search import SearchManager


class Base(models.Model):
    """Abstract base model for all CLX models."""

    id = ShortUUIDField(primary_key=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(default=timezone.now)

    objects = SearchManager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)


class Project(Base):
    """Model for projects."""

    name = models.CharField(max_length=255)
    instructions = models.TextField(blank=True, default="")
    autopilot_enabled = models.BooleanField(default=False)
    active_label = models.ForeignKey(
        "Label",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )

    def add_docs(self, docs, **kwargs):
        """Bulk-insert documents using django-postgres-copy.

        Args:
            docs: Either a DataFrame with a 'text' column, a list of strings
                  (text only), or a list of dicts with 'text' and optionally
                  'meta' keys.
        """
        if docs is None:
            return

        # Normalize input
        if isinstance(docs, pd.DataFrame):
            if docs.empty:
                return
            if "text" not in docs.columns:
                raise ValueError(
                    "DataFrame input must include a 'text' column"
                )

            meta_columns = [col for col in docs.columns if col != "text"]
            docs = [
                {
                    "text": row["text"],
                    "meta": {
                        key: value
                        for key, value in row[meta_columns].items()
                        if pd.notna(value)
                    },
                }
                for _, row in docs.iterrows()
            ]
        elif not docs:
            return
        elif isinstance(docs[0], str):
            docs = [{"text": t, "meta": {}} for t in docs]
        else:
            docs = [
                {"text": d["text"], "meta": d.get("meta", {})} for d in docs
            ]

        # Build DataFrame
        data = pd.DataFrame(docs)
        data = data.dropna(subset=["text"])
        if len(data) == 0:
            return
        data["id"] = [uuid() for _ in range(len(data))]
        data["text_prefix"] = data["text"].str[:50]
        data["text_hash"] = data["text"].apply(generate_hash)
        data["shuffle_key"] = [
            random.randint(0, 1_000_000) for _ in range(len(data))
        ]
        data["meta"] = data["meta"].apply(json.dumps)

        # Write to CSV buffer
        f = StringIO()
        data.to_csv(f, index=False)
        f.seek(0)

        # Bulk insert
        print("Pushing documents to database...")
        Document.objects.from_csv(
            f,
            static_mapping={
                "project_id": str(self.id),
                "created_at": timezone.now(),
                "updated_at": timezone.now(),
            },
            ignore_conflicts=True,
            **kwargs,
        )

    @property
    def project_dir(self):
        """Return the project directory path."""
        return Path(CLX_HOME) / "projects" / str(self.id)

    def export_data(self, batch_size=100_000):
        """Export documents to docs.csv in the project directory.

        Only exports documents created since the last export.
        """
        self.project_dir.mkdir(parents=True, exist_ok=True)
        docs_path = self.project_dir / "docs.csv"
        exported_path = self.project_dir / "exported.txt"

        # Check last export time.
        last_export = None
        if exported_path.exists():
            try:
                last_export = datetime.fromisoformat(
                    exported_path.read_text().strip()
                )
            except (ValueError, OSError):
                pass

        qs = self.documents.order_by("created_at")
        if last_export:
            qs = qs.filter(created_at__gt=last_export)

        total = qs.count()
        if total == 0:
            return

        try:
            for offset in tqdm(
                range(0, total, batch_size),
                desc="Exporting docs",
                total=(total + batch_size - 1) // batch_size,
            ):
                batch = qs.values_list("id", "text")[
                    offset : offset + batch_size
                ]
                data = pd.DataFrame(
                    list(batch), columns=["document_id", "text"]
                )
                pd_save_or_append(data, docs_path)
        except BaseException:
            docs_path.unlink(missing_ok=True)
            exported_path.unlink(missing_ok=True)
            raise

        now = datetime.now(UTC).isoformat()
        exported_path.write_text(now)

    def update_tasks(self):
        """Sync tasks based on current project/label state.

        Rules:
        - No project instructions → project_understanding task (no label)
        - Has project instructions but label lacks instructions → label_understanding per label
        - Both have instructions but label has no training examples → sampling_strategy per label
        - Label has unannotated training examples → annotate per label
        """
        from django.db.models import Count, Q

        expected = []  # list of (prompt_id, label_id | None)

        if not self.instructions.strip():
            expected.append(("project_understanding", None))
        else:
            labels = list(self.labels.all())
            for label in labels:
                if not label.instructions.strip():
                    expected.append(("label_understanding", label.id))
                else:
                    ld_stats = LabelDocument.objects.filter(
                        label=label
                    ).aggregate(
                        total=Count("id"),
                        annotated=Count(
                            "id",
                            filter=Q(annotations__source="agent"),
                        ),
                    )
                    if ld_stats["total"] == 0:
                        expected.append(("sampling_strategy", label.id))
                    elif ld_stats["annotated"] < ld_stats["total"]:
                        expected.append(("annotate", label.id))

        expected_set = set(expected)
        existing = {(t.prompt_id, t.label_id): t for t in self.tasks.all()}

        # Delete tasks no longer expected (keep in-progress and awaiting-input)
        keep_statuses = (Task.Status.IN_PROGRESS, Task.Status.AWAITING_INPUT)
        to_delete = [
            t.id
            for key, t in existing.items()
            if key not in expected_set and t.status not in keep_statuses
        ]
        if to_delete:
            Task.objects.filter(id__in=to_delete).delete()

        # Create missing tasks
        to_create = [
            Task(project=self, prompt_id=pid, label_id=lid)
            for pid, lid in expected
            if (pid, lid) not in existing
        ]
        if to_create:
            Task.objects.bulk_create(to_create, ignore_conflicts=True)

        return list(self.tasks.select_related("label").order_by("created_at"))


class Document(Base):
    """Model for documents within a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="documents"
    )
    text = models.TextField()
    text_prefix = models.CharField(max_length=50)
    meta = models.JSONField(default=dict, null=True, blank=True)
    shuffle_key = models.IntegerField()
    text_hash = models.CharField(max_length=64)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "text_hash"],
                name="document_project_text_hash_uniq",
            )
        ]
        indexes = [
            models.Index(
                fields=["shuffle_key", "id"],
                name="shuffle_key_idx",
            ),
            models.Index(
                fields=["text_prefix"],
                name="text_prefix_idx",
                opclasses=["text_pattern_ops"],
            ),
            GinIndex(
                fields=["text"],
                name="text_trgm_idx",
                opclasses=["gin_trgm_ops"],
            ),
        ]


class Label(Base):
    """Model for labels within a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="labels"
    )
    name = models.CharField(max_length=255)
    instructions = models.TextField(blank=True, default="")
    autopilot_thread = models.ForeignKey(
        "Thread",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    finetune_id = models.CharField(max_length=255, blank=True, default="")
    finetune_training_args = models.JSONField(default=dict, blank=True)
    finetuned_at = models.DateTimeField(null=True, blank=True)
    finetune_status = models.CharField(max_length=20, blank=True, default="")
    predicted_at = models.DateTimeField(null=True, blank=True)
    prediction_stats = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="label_project_name_uniq",
            )
        ]

    def finetune(self, training_args=None):
        """Kick off a remote finetuning job for this label."""
        import os
        import random

        import pandas as pd
        import requests

        from clx.utils import S3

        training_args = training_args or {}
        self.finetune_training_args = training_args
        self.finetune_status = "pending"
        self.save(
            update_fields=[
                "finetune_training_args",
                "finetune_status",
                "updated_at",
            ]
        )

        # Assemble data from annotated label documents (yes/no only).
        # Single query: join through to document text and annotation value.
        rows = list(
            LabelDocument.objects.filter(
                label=self,
                annotations__source="agent",
                annotations__value__in=["yes", "no"],
            ).values_list(
                "document__text",
                "annotations__value",
            )
        )
        rows = [{"text": text, "label": value} for text, value in rows]

        random.shuffle(rows)
        df = pd.DataFrame(rows)
        split = max(1, int(len(df) * 0.2))
        eval_data = df.iloc[:split]
        train_data = df.iloc[split:]

        # Build training args with sensible defaults.
        import math

        from clx.ml import training_run

        batch_size = training_args.get("per_device_train_batch_size", 8)
        grad_accum = training_args.get("gradient_accumulation_steps", 1)
        effective_batch = batch_size * grad_accum
        steps_per_epoch = max(1, math.ceil(len(train_data) / effective_batch))
        checkpoint_steps = max(1, steps_per_epoch // 5)

        defaults = {
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_train_epochs": 3,
            "eval_strategy": "steps",
            "eval_steps": checkpoint_steps,
            "save_steps": checkpoint_steps,
        }
        merged_args = {**defaults, **training_args}

        run = training_run(
            task="classification",
            run_name=str(self.id),
            label_names=["yes", "no"],
            training_args=merged_args,
        )

        # Upload data to S3 and submit to RunPod.
        import tempfile
        import uuid as _uuid
        from pathlib import Path

        endpoint_id = os.getenv("RUNPOD_FINETUNE_ENDPOINT_ID")
        api_key = os.getenv("RUNPOD_API_KEY")
        if not endpoint_id or not api_key:
            self.finetune_status = "error"
            self.save(update_fields=["finetune_status", "updated_at"])
            raise ValueError(
                "RUNPOD_FINETUNE_ENDPOINT_ID and RUNPOD_API_KEY must be set"
            )

        s3 = S3()
        job_key = str(_uuid.uuid4())
        s3_prefix = f"runpod/finetune/{job_key}"

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            train_data.to_csv(train_path, index=False)
            s3.upload(train_path, f"{s3_prefix}/train.csv")
            eval_path = Path(tmpdir) / "eval.csv"
            eval_data.to_csv(eval_path, index=False)
            s3.upload(eval_path, f"{s3_prefix}/eval.csv")

        config = run.config
        del config["run_dir_parent"]
        payload = {
            "input": {
                "training_run": config,
                "s3_bucket": s3.bucket,
                "s3_prefix": s3_prefix,
                "overwrite": True,
            }
        }

        response = requests.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/run",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        job_id = response.json()["id"]

        self.finetune_id = job_id
        self.finetune_status = "in_progress"
        self.save(
            update_fields=[
                "finetune_id",
                "finetune_status",
                "updated_at",
            ]
        )
        return job_id

    @property
    def pipe(self):
        """Remote classification pipeline for the finetuned model."""
        from clx.ml import pipeline

        model_path = f"/runpod-volume/clx/runs/{self.id}/model"
        return pipeline(
            task="classification",
            model=model_path,
            remote=True,
        )

    def predict(self):
        """Run predictions across all label documents using the finetuned model."""
        from django.utils import timezone

        if self.finetune_status != "completed":
            raise ValueError("No completed finetune available.")

        ld_qs = LabelDocument.objects.filter(label=self).select_related(
            "document"
        )
        ld_list = list(ld_qs.values_list("id", "document__text"))
        if not ld_list:
            return

        ld_ids = [row[0] for row in ld_list]
        texts = [row[1] for row in ld_list]

        results = self.pipe.predict(texts, batch_size=16, return_scores=True)

        # Build predictions map.
        predictions = {}
        for ld_id, scores in zip(ld_ids, results):
            yes_score = scores.get("yes", 0)
            no_score = scores.get("no", 0)
            pred = "yes" if yes_score >= no_score else "no"
            top_score = max(yes_score, no_score)
            confidence = abs(top_score - 0.5) * 2
            predictions[ld_id] = (pred, confidence)

        # Bulk update predictions.
        ld_objs = []
        for ld_id, (pred, confidence) in predictions.items():
            obj = LabelDocument(id=ld_id)
            obj.prediction = pred
            obj.prediction_confidence = confidence
            ld_objs.append(obj)

        LabelDocument.objects.bulk_update(
            ld_objs,
            ["prediction", "prediction_confidence"],
            batch_size=1000,
        )

        # Compute F1 and accuracy on annotated examples (yes/no only).
        annotated = dict(
            ClassificationAnnotation.objects.filter(
                label_document_id__in=ld_ids,
                source="agent",
                value__in=["yes", "no"],
            ).values_list("label_document_id", "value")
        )

        if annotated:
            tp = fp = fn = correct = 0
            total = len(annotated)
            for ld_id, true_val in annotated.items():
                pred_val = predictions[ld_id][0]
                if pred_val == true_val:
                    correct += 1
                if pred_val == "yes" and true_val == "yes":
                    tp += 1
                elif pred_val == "yes" and true_val == "no":
                    fp += 1
                elif pred_val == "no" and true_val == "yes":
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            self.prediction_stats = {
                "f1": round(f1, 4),
                "accuracy": round(correct / total, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "total": total,
            }
        else:
            self.prediction_stats = {}

        self.predicted_at = timezone.now()
        self.save(
            update_fields=[
                "predicted_at",
                "prediction_stats",
                "updated_at",
            ]
        )

    def recalculate_prediction_stats(self):
        """Recompute F1/accuracy from existing predictions vs annotations."""
        ld_data = list(
            LabelDocument.objects.filter(
                label=self,
            )
            .exclude(prediction="")
            .filter(prediction__isnull=False)
            .values_list("id", "prediction")
        )
        if not ld_data:
            self.prediction_stats = {}
            self.save(update_fields=["prediction_stats", "updated_at"])
            return

        predictions = dict(ld_data)
        ld_ids = list(predictions.keys())

        annotated = dict(
            ClassificationAnnotation.objects.filter(
                label_document_id__in=ld_ids,
                source="agent",
                value__in=["yes", "no"],
            ).values_list("label_document_id", "value")
        )

        if annotated:
            tp = fp = fn = correct = 0
            total = len(annotated)
            for ld_id, true_val in annotated.items():
                pred_val = predictions.get(ld_id)
                if not pred_val:
                    continue
                if pred_val == true_val:
                    correct += 1
                if pred_val == "yes" and true_val == "yes":
                    tp += 1
                elif pred_val == "yes" and true_val == "no":
                    fp += 1
                elif pred_val == "no" and true_val == "yes":
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            self.prediction_stats = {
                "f1": round(f1, 4),
                "accuracy": round(correct / total, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "total": total,
            }
        else:
            self.prediction_stats = {}

        self.save(update_fields=["prediction_stats", "updated_at"])


class Prompt(Base):
    """A customizable prompt template for a project."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="prompts"
    )
    prompt_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    content = models.TextField(blank=True, default="")
    built_in = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "prompt_id"],
                name="prompt_project_promptid_uniq",
            )
        ]


class Thread(Base):
    """Model for LLM threads tied to a label."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="threads"
    )
    model = models.CharField(
        max_length=255, default=django_settings.DEFAULT_MODEL
    )
    state = models.JSONField(default=dict, blank=True)
    total_cost = models.FloatField(default=0.0)
    autopilot_locked = models.BooleanField(default=False)


class LabelDocument(Base):
    """Links a document to a label (e.g. as a training example)."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="label_documents"
    )
    document = models.ForeignKey(
        "Document", on_delete=models.CASCADE, related_name="label_documents"
    )
    prediction = models.CharField(
        max_length=3,
        blank=True,
        default="",
        choices=[("yes", "yes"), ("no", "no")],
    )
    prediction_confidence = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["label", "document"],
                name="labeldocument_label_document_uniq",
            )
        ]


class ClassificationAnnotation(Base):
    """An annotation on a label document."""

    class Value(models.TextChoices):
        YES = "yes"
        NO = "no"
        SKIP = "skip"

    label_document = models.ForeignKey(
        LabelDocument, on_delete=models.CASCADE, related_name="annotations"
    )
    value = models.CharField(max_length=4, choices=Value.choices)
    source = models.CharField(max_length=255)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["label_document", "source"],
                name="annotation_labeldoc_source_uniq",
            )
        ]


class Task(Base):
    """A pending task for a project (e.g. 'annotate label X')."""

    class Status(models.TextChoices):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        AWAITING_INPUT = "awaiting_input"

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="tasks"
    )
    prompt_id = models.CharField(max_length=255)
    label = models.ForeignKey(
        Label,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tasks",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "prompt_id", "label"],
                name="task_project_prompt_label_uniq",
            )
        ]


class Message(Base):
    """Model for messages within a thread."""

    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="messages"
    )
    data = models.JSONField(default=dict)
    num_tokens = models.IntegerField(default=0)
    is_compact = models.BooleanField(default=False)
    hidden = models.BooleanField(default=False)
