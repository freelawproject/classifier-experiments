import lmdb
import numpy as np
import pandas as pd
import simplejson as json
from django.apps import apps
from django.db import models
from django.utils import timezone
from tqdm import tqdm

from clx import label2slug
from clx.llm import GEPAPredictor, SingleLabelPredictor, batch_embed, mesh_sort
from clx.ml import pipeline, training_run
from clx.settings import CLX_HOME
from clx.utils import pd_save_or_append

from .custom_heuristics import custom_heuristics
from .search_utils import BaseModel, SearchDocumentModel


class Project(BaseModel):
    """Model for projects."""

    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255)
    model_name = models.CharField(max_length=255, unique=True)
    tags_model_name = models.CharField(max_length=255, null=True, blank=True)
    instructions = models.TextField(null=True, blank=True)

    @property
    def data_dir(self):
        return CLX_HOME / "app_projects" / self.id

    @property
    def cached_documents_path(self):
        return self.data_dir / "docs.csv"

    @property
    def cached_embeddings_path(self):
        return self.data_dir / "embeddings.lmdb"

    def load_or_add_embeddings(self, data):
        assert all(x in data.columns for x in ["text_hash", "text"])
        db = lmdb.open(str(self.cached_embeddings_path), map_size=1024**4)
        with db.begin() as c:
            data["embedding"] = data["text_hash"].apply(
                lambda x: c.get(x.encode("utf-8"))
            )
            data["embedding"] = data["embedding"].apply(
                lambda x: json.loads(x) if x is not None else None
            )
        needs_embeddings = data[data["embedding"].isna()]
        data = data[data["embedding"].notna()]
        needs_embeddings["embedding"] = batch_embed(
            needs_embeddings["text"].tolist(),
            num_workers=16,
            dimensions=96,
        )
        with db.begin(write=True) as c:
            for row in needs_embeddings.to_dict("records"):
                c.put(
                    row["text_hash"].encode("utf-8"),
                    json.dumps(row["embedding"]).encode("utf-8"),
                )
        data = pd.concat([data, needs_embeddings])
        return data

    @property
    def cached_documents(self):
        return pd.read_csv(self.cached_documents_path)

    def get_search_model(self):
        """Get the search model class for the project."""
        return apps.get_model("app", self.model_name)

    def get_tags_model(self):
        """Get the tags model class for the project."""
        return apps.get_model("app", self.tags_model_name)


class Label(BaseModel):
    """Model for labels."""

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="labels"
    )
    name = models.CharField(max_length=255)

    # Sample counts
    num_excluded = models.IntegerField(default=0)
    num_neutral = models.IntegerField(default=0)
    num_likely = models.IntegerField(default=0)

    # Predictor config
    llm_models = [
        ("GPT-5 Mini", "openai/gpt-5-mini"),
        ("GPT-5", "openai/gpt-5"),
        ("Gemini 2.5 Flash Lite", "gemini/gemini-2.5-flash-lite"),
        ("Gemini 2.5 Flash", "gemini/gemini-2.5-flash"),
        ("Gemini 2.5 Pro", "gemini/gemini-2.5-pro"),
        ("Qwen 235B-A22B", "bedrock/qwen.qwen3-235b-a22b-2507-v1:0"),
        (
            "Claude Sonnet 4.5",
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
    ]
    default_inference_model = "openai/gpt-5-mini"
    default_teacher_model = "openai/gpt-5"
    instructions = models.TextField(null=True, blank=True)
    inference_model = models.CharField(
        max_length=255,
        choices=llm_models,
        default=default_inference_model,
    )
    teacher_model = models.CharField(
        max_length=255,
        choices=llm_models,
        default=default_teacher_model,
    )
    predictor_data = models.JSONField(null=True, blank=True)
    predictor_updated_at = models.DateTimeField(null=True, blank=True)

    # Trainset config
    trainset_examples_per_heuristic_bucket = models.IntegerField(default=1000)
    trainset_num_excluded = models.IntegerField(default=1000)
    trainset_num_neutral = models.IntegerField(default=1000)
    trainset_num_likely = models.IntegerField(default=1000)
    trainset_num_decision_neighbors = models.IntegerField(default=50)
    trainset_updated_at = models.DateTimeField(null=True, blank=True)
    trainset_predictions_updated_at = models.DateTimeField(
        null=True, blank=True
    )
    trainset_num_positive_preds = models.IntegerField(default=0)
    trainset_num_negative_preds = models.IntegerField(default=0)

    @property
    def data_dir(self):
        return self.project.data_dir / "labels" / f"{label2slug(self.name)}"

    def excluded_query(self, queryset=None):
        if queryset is None:
            queryset = self.project.get_search_model().objects
        tags = LabelTag.objects.filter(label=self, heuristic__is_minimal=True)
        tag_ids = tags.values_list("id", flat=True)
        if not tag_ids:
            return queryset.none()
        return queryset.tags(not_any=tag_ids)

    def neutral_query(self, queryset=None):
        if queryset is None:
            queryset = self.project.get_search_model().objects
        minimal_tags = LabelTag.objects.filter(
            label=self, heuristic__is_minimal=True
        )
        minimal_tag_ids = minimal_tags.values_list("id", flat=True)
        likely_tags = LabelTag.objects.filter(
            label=self, heuristic__is_likely=True
        )
        likely_tag_ids = likely_tags.values_list("id", flat=True)
        return queryset.tags(any=minimal_tag_ids, not_any=likely_tag_ids)

    def likely_query(self, queryset=None):
        if queryset is None:
            queryset = self.project.get_search_model().objects
        minimal_tags = LabelTag.objects.filter(
            label=self, heuristic__is_minimal=True
        )
        minimal_tag_ids = minimal_tags.values_list("id", flat=True)
        likely_tags = LabelTag.objects.filter(
            label=self, heuristic__is_likely=True
        )
        likely_tag_ids = likely_tags.values_list("id", flat=True)
        if not likely_tag_ids:
            return queryset.none()
        return queryset.tags(any=minimal_tag_ids).tags(any=likely_tag_ids)

    def update_counts(self):
        self.num_excluded = self.excluded_query().count()
        self.num_likely = self.likely_query().count()
        self.num_neutral = self.neutral_query().count()
        self.save()

    def sample_trainset(self, ratio=1):
        """Sample trainset examples."""
        data = []
        # Sample decision neighbors
        model = self.project.get_search_model()
        for decision in self.decisions.all():
            embedding = (
                model.objects.filter(text_hash=decision.text_hash)
                .first()
                .embedding.to_list()
            )
            decision_examples = model.objects.search(
                semantic_sort=embedding,
                page_size=int(self.trainset_num_decision_neighbors * ratio),
            )
            data += [{"id": x["id"]} for x in decision_examples["data"]]

        def apply_mesh_sort(queryset, n_examples):
            """Select 10x the number of examples and take most diverse 10%"""
            cluster_ks = [10, 10]
            data = queryset.order_by("?").values("id", "embedding")
            data = pd.DataFrame(data[: n_examples * 10])
            data["embedding"] = data["embedding"].apply(lambda x: x.to_list())
            data["sort"] = mesh_sort(
                np.array(data["embedding"].tolist()), cluster_ks
            )
            data = data.sort_values(by="sort").head(n_examples)
            return data[["id"]].to_dict("records")

        # Sample heuristic buckets
        data += apply_mesh_sort(
            self.excluded_query(), int(self.trainset_num_excluded * ratio)
        )
        data += apply_mesh_sort(
            self.neutral_query(), int(self.trainset_num_neutral * ratio)
        )
        data += apply_mesh_sort(
            self.likely_query(), int(self.trainset_num_likely * ratio)
        )

        data = pd.DataFrame(data).drop_duplicates(subset="id").sample(frac=1)
        return data["id"].tolist()

    def update_trainset(self):
        self.trainset_examples.all().delete()
        model = self.project.get_search_model()

        train_ids = self.sample_trainset(ratio=1)
        train_examples = model.objects.filter(id__in=train_ids).values(
            "text", "text_hash"
        )
        train_examples = pd.DataFrame(train_examples)
        train_examples["split"] = "train"

        eval_ids = self.sample_trainset(ratio=0.2)
        eval_examples = model.objects.filter(id__in=eval_ids).values(
            "text", "text_hash"
        )
        eval_examples = pd.DataFrame(eval_examples)
        eval_examples["split"] = "eval"

        trainset = pd.concat([train_examples, eval_examples])
        trainset = trainset.drop_duplicates(subset="text_hash")
        rows = trainset.to_dict("records")
        LabelTrainsetExample.objects.bulk_create(
            [LabelTrainsetExample(label_id=self.id, **row) for row in rows],
            batch_size=1000,
        )
        self.sync_trainset_tags()
        self.trainset_updated_at = timezone.now()
        self.save()

    def load_annos(self):
        project = self.project
        search_model = project.get_search_model()

        pos_annos = search_model.objects.tags(
            any=[self.anno_true_tag.id]
        ).values("text_hash", "text")
        pos_annos = pd.DataFrame(pos_annos)
        pos_annos["value"] = True

        neg_annos = search_model.objects.tags(
            any=[self.anno_false_tag.id]
        ).values("text_hash", "text")
        neg_annos = pd.DataFrame(neg_annos)
        neg_annos["value"] = False

        flag_annos = search_model.objects.tags(
            any=[self.anno_flag_tag.id]
        ).values("text_hash", "text")
        flag_annos = pd.DataFrame(flag_annos)
        flag_annos["value"] = None

        annos = pd.concat([pos_annos, neg_annos, flag_annos])
        if annos.empty:
            return pd.DataFrame(columns=["text_hash", "text", "value"])
        return annos

    def load_trainset(self):
        data = pd.DataFrame(
            self.trainset_examples.all().values(
                "text_hash", "text", "split", "pred", "reason"
            )
        )

        annos = self.load_annos()
        flagged_hashes = annos[annos["value"].isna()]["text_hash"].tolist()
        annos = annos[~annos["value"].isna()]
        annos = annos.rename(columns={"value": "pred"})
        annos["split"] = "train"
        data = pd.concat([data, annos])

        if len(data) and "text_hash" in data.columns:
            data = data.drop_duplicates(subset="text_hash", keep="last")
            data = data[~data["text_hash"].isin(flagged_hashes)]

        data = data.sample(frac=1, random_state=42)
        data = data.reset_index(drop=True)
        return data

    def update_trainset_preds(self, num_threads=128):
        predictor = self.predictor
        trainset = self.load_trainset()
        preds = predictor.predict(
            trainset["text"].tolist(), num_threads=num_threads
        )
        trainset["pred"] = [x.value for x in preds]
        trainset["reason"] = [x.reason for x in preds]
        examples = self.trainset_examples.all()
        examples = {e.text_hash: e for e in examples}
        for row in trainset.to_dict("records"):
            if row["text_hash"] in examples:
                example = examples[row["text_hash"]]
                example.pred = row["pred"]
                example.reason = row["reason"]
        LabelTrainsetExample.objects.bulk_update(
            list(examples.values()),
            fields=["pred", "reason"],
            batch_size=1000,
        )
        self.update_trainset_pred_counts()
        self.sync_trainset_pred_tags()
        self.trainset_predictions_updated_at = timezone.now()
        self.save()

    def update_trainset_pred_counts(self):
        data = self.load_trainset()
        if len(data) and "pred" in data.columns:
            data = data.dropna(subset=["pred"])
            preds = data["pred"].astype(bool)
            self.trainset_num_positive_preds = preds.sum()
            self.trainset_num_negative_preds = (~preds).sum()
        else:
            self.trainset_num_positive_preds = 0
            self.trainset_num_negative_preds = 0
        self.save()

    def get_new_predictor(self):
        return SingleLabelPredictor(
            label_name=self.name,
            project_instructions=self.project.instructions,
            label_instructions=self.instructions,
            model=self.inference_model,
        )

    @property
    def predictor(self):
        if self.predictor_data is None:
            return self.get_new_predictor()
        else:
            return GEPAPredictor.from_config(self.predictor_data)

    @property
    def trainset_train_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="trainset:train",
            label=self,
        )
        return tag

    @property
    def trainset_eval_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="trainset:eval",
            label=self,
        )
        return tag

    @property
    def trainset_pred_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="trainset:pred",
            label=self,
        )
        return tag

    def get_trainset_finetune_tag(self, config_name):
        tag, _ = LabelTag.objects.get_or_create(
            name=f"trainset:ft:{config_name}",
            label=self,
        )
        return tag

    @property
    def finetune_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="ft",
            label=self,
        )
        return tag

    @property
    def anno_true_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="anno:true",
            label=self,
        )
        return tag

    @property
    def anno_false_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="anno:false",
            label=self,
        )
        return tag

    @property
    def anno_flag_tag(self):
        tag, _ = LabelTag.objects.get_or_create(
            name="anno:flag",
            label=self,
        )
        return tag

    def sync_trainset_tags(self):
        """Sync tags for train/eval splits to match current trainset examples."""
        model = self.project.get_search_model()

        train_hashes = list(
            self.trainset_examples.filter(split="train").values_list(
                "text_hash", flat=True
            )
        )
        if train_hashes:
            train_ids = list(
                model.objects.filter(text_hash__in=train_hashes).values_list(
                    "id", flat=True
                )
            )
        else:
            train_ids = []
        model.bulk_replace_tag(self.trainset_train_tag, train_ids)

        eval_hashes = list(
            self.trainset_examples.filter(split="eval").values_list(
                "text_hash", flat=True
            )
        )
        if eval_hashes:
            eval_ids = list(
                model.objects.filter(text_hash__in=eval_hashes).values_list(
                    "id", flat=True
                )
            )
        else:
            eval_ids = []
        model.bulk_replace_tag(self.trainset_eval_tag, eval_ids)

    def sync_trainset_pred_tags(self):
        """Sync tag for positive predictions to match current predicted positives."""
        model = self.project.get_search_model()
        pos_hashes = list(
            self.trainset_examples.filter(pred=True).values_list(
                "text_hash", flat=True
            )
        )
        if pos_hashes:
            pos_ids = list(
                model.objects.filter(text_hash__in=pos_hashes).values_list(
                    "id", flat=True
                )
            )
        else:
            pos_ids = []
        model.bulk_replace_tag(self.trainset_pred_tag, pos_ids)

    def fit_predictor(self):
        predictor = self.get_new_predictor()
        examples = self.decisions.values("text", "value", "reason")
        predictor.fit(
            examples,
            num_threads=8,
            reflection_lm={
                "model": self.teacher_model,
                "temperature": 1.0,
                "max_tokens": 32000,
            },
        )
        self.predictor_data = predictor.config
        self.predictor_updated_at = timezone.now()
        self.save()
        print(predictor.last_cost)

    def get_finetune_run_name(self, config_name):
        return f"{self.project_id}__{label2slug(self.name)}__{config_name}"

    def get_finetune_run_pipe(self, config_name):
        run_name = self.get_finetune_run_name(config_name)
        model_path = f"/runpod-volume/clx/runs/{run_name}/model"
        return pipeline(task="classification", model=model_path, remote=True)

    def prepare_finetune(
        self, config_name, batch_size=16, gradient_accumulation_steps=1
    ):
        model = self.project.get_search_model()
        config = model.finetune_configs[config_name]
        data = self.load_trainset()
        data = data.sample(frac=1, random_state=42)
        data = (
            data[["text_hash", "text", "pred", "split"]]
            .rename(columns={"pred": "label"})
            .dropna()
        )
        data["label"] = data["label"].apply(lambda x: "yes" if x else "no")
        train_data = data[data["split"] == "train"]
        eval_data = data[data["split"] == "eval"]

        num_train_epochs = config["training_args"].get("num_train_epochs", 1)
        config["training_args"]["num_train_epochs"] = num_train_epochs
        total_steps = (num_train_epochs * len(train_data)) // (
            batch_size * gradient_accumulation_steps
        )
        save_steps = total_steps // 9
        config["training_args"]["eval_strategy"] = "steps"
        config["training_args"]["save_strategy"] = "steps"
        config["training_args"]["eval_steps"] = save_steps
        config["training_args"]["save_steps"] = save_steps
        config["training_args"]["per_device_train_batch_size"] = batch_size
        config["training_args"]["per_device_eval_batch_size"] = batch_size
        config["training_args"]["gradient_accumulation_steps"] = (
            gradient_accumulation_steps
        )

        run_config = {
            "task": "classification",
            "run_name": self.get_finetune_run_name(config_name),
            "label_names": ["yes", "no"],
            **config,
        }

        return train_data, eval_data, run_config

    def train_finetune(self, config_name):
        """Train a finetune model for this label."""
        train_data, eval_data, run_config = self.prepare_finetune(config_name)

        run = training_run(**run_config)
        outputs = run.train(train_data, eval_data, overwrite=True, remote=True)

        data = pd.concat([train_data, eval_data])

        pipe = self.get_finetune_run_pipe(config_name)
        data["pred"] = pipe(data["text"].tolist(), batch_size=16)
        data = data[data["pred"] == "yes"]

        tag = self.get_trainset_finetune_tag(config_name)
        model = self.project.get_search_model()
        example_ids = model.objects.filter(
            text_hash__in=data["text_hash"].tolist()
        )
        example_ids = example_ids.values_list("id", flat=True)
        model.bulk_replace_tag(tag.id, example_ids)

        finetune, _ = LabelFinetune.objects.get_or_create(
            label=self, config_name=config_name
        )
        finetune.eval_results = outputs["results"]
        finetune.finetuned_at = timezone.now()
        finetune.save()

        return finetune

    def predict_finetune(self, batch_size=16, num_workers=64, force=False):
        """Run finetune predictions across the entire corpus."""
        cache_path = self.data_dir / "finetune_predictions_cache.csv"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_name = self.project.get_search_model().main_finetune_config
        if config_name is None:
            raise ValueError("Set main_finetune_config for this project")

        if force and cache_path.exists():
            cache_path.unlink()

        cached_ids = set()
        if cache_path.exists():
            cached_data = pd.read_csv(cache_path)
            cached_ids = set(cached_data["id"].unique().tolist())

        model = self.project.get_search_model()
        pipe = self.get_finetune_run_pipe(config_name)

        minimal_heuristics = LabelHeuristic.objects.filter(
            is_minimal=True, label=self
        )
        minimal_conditions = [h.get_apply_fn() for h in minimal_heuristics]

        def minimal_condition_fn(text):
            return any(condition(text) for condition in minimal_conditions)

        total_examples = model.objects.count()
        outer_batch_size = 1024 * 500
        for batch in tqdm(
            model.objects.batch_df("id", "text", batch_size=outer_batch_size),
            desc=f"Predicting {config_name}",
            total=total_examples // outer_batch_size,
        ):
            batch = batch[~batch["id"].isin(cached_ids)]
            batch = batch[batch["text"].apply(minimal_condition_fn)]
            if len(batch) > 0:
                batch["value"] = pipe(
                    batch["text"].tolist(),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_length=768,
                    truncation=True,
                )
                batch["value"] = batch["value"].apply(lambda x: x == "yes")
                pd_save_or_append(batch[["id", "value"]], cache_path)

        if cache_path.exists():
            all_preds = pd.read_csv(cache_path)
            positive_ids = all_preds[all_preds["value"]]["id"].tolist()
            tag = self.finetune_tag
            model.bulk_replace_tag(tag.id, positive_ids)
            finetune = self.fintunes.filter(config_name=config_name).first()
            if finetune:
                finetune.predicted_at = timezone.now()
                finetune.save()
            cache_path.unlink()

            print(
                f"Predictions complete: {len(positive_ids):,} positive out of {len(all_preds):,} total"
            )

    def update_all(self, num_threads=128, predict=False, force=False):
        """Update all components that are out of date based on timestamps.

        Runs the full pipeline in order, but only steps that need updating:
        1. Resample trainset (if decisions newer than trainset)
        2. Fit predictor (if trainset newer than predictor)
        3. Run predictions (if predictor newer than predictions)
        4. Train finetunes (if predictions newer than finetunes)
        5. Run global corpus predictions (if predict is True and finetune newer than global predictions)
        """
        missing = []
        if not self.heuristics.filter(is_minimal=True).exists():
            missing.append("at least one minimal heuristic")
        if not self.heuristics.filter(is_likely=True).exists():
            missing.append("at least one likely heuristic")
        if not self.decisions.filter(value=True).exists():
            missing.append("at least one positive decision")
        if not self.decisions.filter(value=False).exists():
            missing.append("at least one negative decision")

        if missing:
            print("Cannot run update_all - missing required setup:")
            for item in missing:
                print(f"  - {item}")
            return

        model = self.project.get_search_model()
        finetune_configs = list(model.finetune_configs.keys())

        # Get latest decision timestamp
        latest_decision = self.decisions.order_by("-updated_at").first()
        latest_decision_at = (
            latest_decision.updated_at if latest_decision else None
        )

        # Step 1: Resample trainset if decisions are newer
        if force or (
            latest_decision_at
            and (
                not self.trainset_updated_at
                or latest_decision_at > self.trainset_updated_at
            )
        ):
            print("Step 1: Resampling trainset")
            self.update_trainset()
            self.refresh_from_db()

        # Step 2: Fit predictor if trainset is newer
        if force or (
            self.trainset_updated_at
            and (
                not self.predictor_updated_at
                or self.trainset_updated_at > self.predictor_updated_at
            )
        ):
            print("Step 2: Fitting predictor")
            self.fit_predictor()
            self.refresh_from_db()

        # Step 3: Run predictions if predictor is newer
        if force or (
            self.predictor_updated_at
            and (
                not self.trainset_predictions_updated_at
                or self.predictor_updated_at
                > self.trainset_predictions_updated_at
            )
        ):
            print("Step 3: Running predictions")
            self.update_trainset_preds(num_threads=num_threads)
            self.refresh_from_db()

        # Step 4: Train finetunes if predictions are newer
        for config_name in finetune_configs:
            finetune = self.fintunes.filter(config_name=config_name).first()
            finetuned_at = finetune.finetuned_at if finetune else None

            if force or (
                self.trainset_predictions_updated_at
                and (
                    not finetuned_at
                    or self.trainset_predictions_updated_at > finetuned_at
                )
            ):
                print(f"Step 4: Training finetune: {config_name}")
                self.train_finetune(config_name)

        # Step 5: Run global corpus predictions if finetune is newer
        if predict:
            ft = self.fintunes.filter(
                config_name=self.project.get_search_model().main_finetune_config
            ).first()
            if ft and (
                force
                or (
                    ft.finetuned_at
                    and (
                        not ft.predicted_at
                        or ft.finetuned_at > ft.predicted_at
                    )
                )
            ):
                print("Step 5: Running global predictions")
                self.predict_finetune(force=force)

        print("Update complete!")

    class Meta:
        unique_together = ("project", "name")


class LabelTag(BaseModel):
    """Model for label tags."""

    name = models.CharField(max_length=255)
    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="tags"
    )
    slug = models.CharField(max_length=255)
    heuristic = models.OneToOneField(
        "LabelHeuristic",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tag",
    )

    def save(self, *args, **kwargs):
        self.slug = label2slug(self.name) + ":" + label2slug(self.label.name)
        super().save(*args, **kwargs)

    class Meta:
        unique_together = ("name", "label")


class LabelDecision(BaseModel):
    """Model for label decision boundaries."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="decisions"
    )
    text_hash = models.CharField(max_length=255)
    text = models.TextField(null=True, blank=True)
    value = models.BooleanField()
    reason = models.TextField()

    class Meta:
        unique_together = ("label", "text_hash")


class LabelHeuristic(BaseModel):
    """Model for label heuristics."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="heuristics"
    )
    querystring = models.TextField(null=True, blank=True)
    custom = models.CharField(max_length=255, null=True, blank=True)
    applied_at = models.DateTimeField(null=True, blank=True)
    is_minimal = models.BooleanField(default=False)
    is_likely = models.BooleanField(default=False)
    num_examples = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        if sum([bool(self.querystring), bool(self.custom)]) != 1:
            raise ValueError(
                "Exactly one of querystring or custom must be provided."
            )
        super().save(*args, **kwargs)
        if self.applied_at is not None:
            self.label.update_counts()

    def delete(self, *args, **kwargs):
        self.is_minimal = False
        self.is_likely = False
        self.save()
        self.label.update_counts()
        super().delete(*args, **kwargs)

    @property
    def name(self):
        if self.querystring is not None:
            return f"h:qs:{self.querystring}"
        elif self.custom is not None:
            return f"h:fn:{self.custom}"

    @classmethod
    def sync_custom_heuristics(cls):
        for heuristic in cls.objects.filter(custom__isnull=False):
            label = heuristic.label
            if (
                heuristic.custom not in custom_heuristics
                or label.name
                != custom_heuristics[heuristic.custom]["label_name"]
                or label.project_id
                != custom_heuristics[heuristic.custom]["project_id"]
            ):
                heuristic.delete()

        for custom_name, custom_heuristic in custom_heuristics.items():
            heuristic_exists = cls.objects.filter(
                label__name=custom_heuristic["label_name"],
                label__project_id=custom_heuristic["project_id"],
                custom=custom_name,
            ).exists()
            if not heuristic_exists:
                label, _ = Label.objects.get_or_create(
                    name=custom_heuristic["label_name"],
                    project_id=custom_heuristic["project_id"],
                )
                heuristic = cls.objects.create(
                    label=label,
                    custom=custom_name,
                )

    def get_apply_fn(self, **kwargs):
        def apply_fn(text):
            if self.querystring is not None:
                text = text.lower()
                querystring = self.querystring.lower()

                for and_part in querystring.split(","):
                    and_part = and_part.strip()
                    meets_any_or = False
                    for or_part in and_part.split("|"):
                        or_part = or_part.strip()
                        negated = False
                        if or_part.startswith("~"):
                            or_part = or_part[1:].strip()
                            negated = True
                        if or_part.startswith("^"):
                            or_part = or_part[1:].strip()
                            if text.startswith(or_part.strip()) != negated:
                                meets_any_or = True
                        elif (or_part.strip() in text) != negated:
                            meets_any_or = True
                    if not meets_any_or:
                        return False
                return True
            elif self.custom is not None:
                return custom_heuristics[self.custom]["apply_fn"](
                    text, **kwargs
                )

        return apply_fn

    def apply(self):
        tag, _ = LabelTag.objects.get_or_create(
            name=self.name, label=self.label, heuristic=self
        )
        apply_fn = self.get_apply_fn()
        example_ids = []
        model = self.label.project.get_search_model()
        batch_size = 1000000
        batches = model.objects.batch_df("id", "text", batch_size=batch_size)
        for batch in tqdm(
            batches,
            desc="Applying heuristic",
            total=model.objects.count() // batch_size,
        ):
            batch = batch[batch["text"].apply(apply_fn)]
            example_ids.extend(batch["id"].tolist())
        model.bulk_replace_tag(tag.id, example_ids)
        self.applied_at = timezone.now()
        self.num_examples = model.objects.tags(any=[tag.id]).count()
        self.save()
        self.label.update_counts()


class LabelTrainsetExample(BaseModel):
    """Model for label trainset examples."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="trainset_examples"
    )
    text_hash = models.CharField(max_length=255)
    text = models.TextField(null=True, blank=True)
    split = models.CharField(
        max_length=10,
        choices=[("train", "Train"), ("eval", "Eval")],
    )
    pred = models.BooleanField(null=True, blank=True)
    reason = models.TextField(null=True, blank=True)

    class Meta:
        unique_together = ("label", "text_hash")


class LabelFinetune(BaseModel):
    """Model for single-label finetuned models."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="fintunes"
    )
    config_name = models.CharField(max_length=255)
    eval_results = models.JSONField(null=True, blank=True)
    finetuned_at = models.DateTimeField(null=True, blank=True)
    predicted_at = models.DateTimeField(null=True, blank=True)


class DocketEntry(SearchDocumentModel):
    """Docket entry model for main document entries."""

    project_id = "docket-entry"
    finetune_configs = {
        "underfit": {
            "base_model_name": "answerdotai/ModernBERT-base",
            "training_args": {
                "num_train_epochs": 1,
                "learning_rate": 5e-5,
                "warmup_ratio": 0.05,
                "bf16": True,
            },
        },
        "main": {
            "base_model_name": "answerdotai/ModernBERT-base",
            "training_args": {
                "num_train_epochs": 10,
                "learning_rate": 5e-5,
                "warmup_ratio": 0.05,
                "bf16": True,
            },
        },
    }
    main_finetune_config = "main"

    id = models.BigIntegerField(primary_key=True)
    recap_id = models.BigIntegerField(unique=True)
    docket_id = models.BigIntegerField()
    entry_number = models.BigIntegerField(null=True, blank=True)
    date_filed = models.DateField(null=True, blank=True)


DocketEntry.create_tags_model()


class DocketEntryShort(SearchDocumentModel):
    """Model for attachments and docket entry short descriptions."""

    project_id = "docket-entry-short"

    text = models.TextField(unique=True)
    text_type = models.CharField(
        max_length=255,
        choices=[
            ("short_description", "Short Description"),
            ("attachment", "Attachment"),
        ],
    )
    count = models.IntegerField(default=0)


DocketEntryShort.create_tags_model()
