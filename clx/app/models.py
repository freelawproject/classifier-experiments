from django.apps import apps
from django.db import models
from django.db.models import JSONField, Max
from django.template.defaultfilters import slugify

from .search_utils import BaseModel, SearchDocumentModel


class Project(BaseModel):
    """Model for projects."""

    name = models.CharField(max_length=255)
    model_name = models.CharField(max_length=255, unique=True)
    slug = models.CharField(max_length=255, unique=True)
    instructions = models.TextField(null=True, blank=True)

    def get_search_model_class(self):
        """Get the search model class for the project."""
        return apps.get_model("app", self.model_name)


class Label(BaseModel):
    """Model for labels."""

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    instructions = models.TextField(null=True, blank=True)
    needs_predictor_update = models.BooleanField(default=True)

    class Meta:
        unique_together = ("project", "name")


class LabelFeature(BaseModel):
    """Model for label features."""

    name = models.CharField(max_length=255)
    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="features"
    )
    slug = models.CharField(max_length=255)

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name) + ":" + slugify(self.label.name)
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

    id = models.BigIntegerField(primary_key=True)
    recap_id = models.BigIntegerField(unique=True)
    docket_id = models.BigIntegerField()
    entry_number = models.BigIntegerField(null=True, blank=True)
    date_filed = models.DateField(null=True, blank=True)


class DocketEntryShort(SearchDocumentModel):
    """Model for attachments and docket entry short descriptions."""

    text = models.TextField(unique=True)
    text_type = models.CharField(
        max_length=255,
        choices=[
            ("short_description", "Short Description"),
            ("attachment", "Attachment"),
        ],
    )
    count = models.IntegerField(default=0)
