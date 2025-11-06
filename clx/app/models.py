from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import models
from django.db.models import JSONField, Max
from django.template.defaultfilters import slugify
from postgres_copy import CopyManager


class BaseModel(models.Model):
    """Base model for all models"""

    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class DocketEntry(BaseModel):
    """Docket entry model for main document entries."""

    id = models.BigIntegerField(primary_key=True)
    recap_id = models.BigIntegerField(unique=True)
    docket_id = models.BigIntegerField()
    entry_number = models.BigIntegerField(null=True, blank=True)
    date_filed = models.DateField(null=True, blank=True)
    text = models.TextField()
    features = ArrayField(models.BigIntegerField(), default=list, blank=True)
    shuffle_sort = models.IntegerField()

    objects = CopyManager()

    class Meta:
        indexes = [
            GinIndex(fields=["features"], name="docket_entry_features_gin"),
        ]


class DocketEntryShortText(BaseModel):
    """Model for attachments and docket entry short descriptions."""

    text = models.TextField(unique=True)
    text_type = models.CharField(
        max_length=255,
        choices=[
            ("short_description", "Short Description"),
            ("attachment", "Attachment"),
        ],
    )
    count = models.IntegerField()
    shuffle_sort = models.IntegerField()

    objects = CopyManager()


class DocketEntryLabel(BaseModel):
    """Model for docket entry labels."""

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)
    scales_analogue = models.CharField(max_length=255, null=True, blank=True)
    needs_program_update = models.BooleanField(default=True)


class DocketEntryFeature(BaseModel):
    """Model for docket entry features."""

    name = models.CharField(max_length=255)
    label = models.ForeignKey(
        DocketEntryLabel, on_delete=models.CASCADE, related_name="features"
    )
    slug = models.CharField(max_length=255)

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name) + ":" + slugify(self.label.name)
        super().save(*args, **kwargs)

    class Meta:
        unique_together = ("name", "label")


class DocketEntryLabelDecision(BaseModel):
    """Model for docket entry label decision boundaries."""

    label = models.ForeignKey(
        DocketEntryLabel, on_delete=models.CASCADE, related_name="decisions"
    )
    entry = models.ForeignKey(
        DocketEntry, on_delete=models.CASCADE, related_name="labels"
    )
    value = models.BooleanField()
    reason = models.TextField()


class DocketEntryLabelProgram(BaseModel):
    """Model for label-specific DSPy programs."""

    label = models.ForeignKey(
        DocketEntryLabel, on_delete=models.CASCADE, related_name="programs"
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
