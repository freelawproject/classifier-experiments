import pandas as pd
from django.apps import apps
from django.db import models
from django.db.models import JSONField, Max
from django.utils import timezone

from clx.settings import CACHED_DATASET_DIR

from .custom_heuristics import custom_heuristics
from .search_utils import BaseModel, SearchDocumentModel


class Project(BaseModel):
    """Model for projects."""

    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255)
    model_name = models.CharField(max_length=255, unique=True)
    tags_model_name = models.CharField(max_length=255, null=True, blank=True)
    instructions = models.TextField(null=True, blank=True)

    def get_search_model(self):
        """Get the search model class for the project."""
        return apps.get_model("app", self.model_name)

    def get_tags_model(self):
        """Get the tags model class for the project."""
        return apps.get_model("app", self.tags_model_name)


class Label(BaseModel):
    """Model for labels."""

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    instructions = models.TextField(null=True, blank=True)
    minimal_heuristic = models.ForeignKey(
        "LabelHeuristic",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="minimal_heuristics",
    )
    needs_predictor_update = models.BooleanField(default=True)

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
        self.slug = (
            self.name.lower().replace(" ", "_")
            + ":"
            + self.label.name.lower().replace(" ", "_")
        )
        super().save(*args, **kwargs)

    class Meta:
        unique_together = ("name", "label")


class LabelDecision(BaseModel):
    """Model for label decision boundaries."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="decisions"
    )
    text_hash = models.CharField(max_length=255)
    value = models.BooleanField()
    reason = models.TextField()


class LabelHeuristic(BaseModel):
    """Model for label heuristics."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="heuristics"
    )
    querystring = models.TextField(null=True, blank=True)
    custom = models.CharField(max_length=255, null=True, blank=True)
    applied_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if sum([bool(self.querystring), bool(self.custom)]) != 1:
            raise ValueError(
                "Exactly one of querystring or custom must be provided."
            )
        super().save(*args, **kwargs)

    @property
    def name(self):
        if self.querystring is not None:
            return f"h:qs:{self.querystring}"
        elif self.custom is not None:
            return f"h:fn:{self.custom}"

    def get_apply_fn(self, **kwargs):
        def apply_fn(text):
            if self.querystring is not None:
                text = text.lower()
                querystring = self.querystring.lower()

                for and_part in querystring.split(","):
                    and_part = and_part.strip()
                    for or_part in and_part.split("|"):
                        or_part = or_part.strip()
                        if or_part.startswith("~"):
                            or_part = or_part[1:]
                            if or_part.strip() in text:
                                return False
                        elif or_part.startswith("^"):
                            or_part = or_part[1:]
                            if not text.startswith(or_part.strip()):
                                return False
                        elif or_part.strip() not in text:
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
        data = pd.read_csv(CACHED_DATASET_DIR / f"{self.label.project.id}.csv")
        apply_fn = self.get_apply_fn()
        data = data[data["text"].progress_apply(apply_fn)]
        example_ids = data["id"].tolist()
        model = self.label.project.get_search_model()
        model.bulk_replace_tag(tag.id, example_ids)
        self.applied_at = timezone.now()
        self.save()


class LabelPredictor(BaseModel):
    """Model for label-specific DSPy predictors."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="predictors"
    )
    model = models.CharField(max_length=255)
    teacher_model = models.CharField(max_length=255)
    data = JSONField()
    version = models.IntegerField(default=0)

    def create(self, *args, **kwargs):
        self.version = (
            self.label.predictors.aggregate(Max("version"))["version__max"] + 1
        )
        self.label.needs_predictor_update = True
        self.label.save()
        return super().create(*args, **kwargs)

    def fit(self):
        """Fit the predictor to the latest decisions."""
        raise NotImplementedError("Not implemented")

    class Meta:
        unique_together = ("label", "version")


class DocketEntry(SearchDocumentModel):
    """Docket entry model for main document entries."""

    project_id = "docket-entry"

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
