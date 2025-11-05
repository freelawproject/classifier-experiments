from django.db import models
from django.db.models import JSONField, Max
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

    entry_id = models.IntegerField(unique=True)
    docket_id = models.IntegerField()
    entry_number = models.IntegerField(null=True, blank=True)
    date_filed = models.DateField()
    text = models.TextField()
    shuffle_sort = models.IntegerField()

    objects = CopyManager()


class DocketEntryShortText(BaseModel):
    """Model for attachments and docket entry short descriptions."""

    text = models.TextField(unique=True)
    count = models.IntegerField()
    shuffle_sort = models.IntegerField()

    objects = CopyManager()


class DocketEntryLabel(BaseModel):
    """Model for docket entry labels."""

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)
    scales_analogue = models.CharField(max_length=255, null=True, blank=True)
    needs_program_update = models.BooleanField(default=True)


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
