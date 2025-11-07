from django.db import models
from django.db.models import JSONField, Max
from django.template.defaultfilters import slugify

from .search_utils import BaseModel, SearchDocumentModel


class Project(BaseModel):
    """Model for projects."""

    table_name = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)


class Label(BaseModel):
    """Model for labels."""

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    needs_program_update = models.BooleanField(default=True)

    class Meta:
        unique_together = ("project", "name")


class Feature(BaseModel):
    """Model for features."""

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


class LabelProgram(BaseModel):
    """Model for label-specific DSPy programs."""

    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="programs"
    )
    model = models.CharField(max_length=255)
    teacher_model = models.CharField(max_length=255)
    program = JSONField()
    version = models.IntegerField(default=0)

    def create(self, *args, **kwargs):
        self.version = (
            self.label.programs.aggregate(Max("version"))["version__max"] + 1
        )
        self.label.needs_program_update = True
        self.label.save()
        return super().create(*args, **kwargs)

    def fit(self):
        """Fit the program to the latest decisions."""
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
