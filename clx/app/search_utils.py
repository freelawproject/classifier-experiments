import csv
import random
from io import StringIO

import pandas as pd
import simplejson as json
from django.apps import apps
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import connection, models, transaction
from django.db.models import Q
from django.utils import timezone
from pgvector.django import (
    CosineDistance,
    HalfVectorField,
    HnswIndex,
)
from postgres_copy import CopyManager, CopyQuerySet
from pydantic import BaseModel as PydanticModel

from clx import generate_hash
from clx.llm import batch_embed


# Pydantic Models
class TagParams(PydanticModel):
    any: list[int] = []
    all: list[int] = []
    not_any: list[int] = []
    not_all: list[int] = []


class SearchParams(PydanticModel):
    heuristic_bucket: str | None = None
    trainset_split: str | None = None
    predictor_value: str | None = None
    annotation_value: str | None = None
    tags: TagParams = TagParams()
    querystring: str | None = None


class SearchQuery(PydanticModel):
    active_label_id: int | None = None
    params: SearchParams = SearchParams()
    sort: list[str] = ["shuffle_sort", "id"]
    semantic_sort: str | list[float] | None = None
    page: int = 1
    page_size: int = 100
    count: bool = False


# QuerySets
class SearchQuerySet(CopyQuerySet):
    """QuerySet for search queries."""

    def batch_df(self, *columns, batch_size=1000):
        last_id = None
        self = self.order_by("id")
        while 1:
            if last_id is not None:
                self = self.filter(id__gt=last_id)
            data = pd.DataFrame(self.values(*columns)[:batch_size])
            if len(data) == 0:
                break
            yield data
            last_id = data["id"].max()

    def querystring(self, value):
        """Apply a querystring to the query.

        For querystrings we will do an exact substring match on terms / phrases.
        Commas will be treated as AND operators.
        Bars will be treated as OR operators.
        Tildes will be treated as NOT operators.
        Carets will be treated as startswith operators.
        We will always assume that ORs are nested in ANDs.
        """
        if value is not None:
            assert isinstance(value, str), "Querystring must be a string"
            and_condition = None
            for and_part in value.split(","):
                or_condition = None
                for or_part in and_part.split("|"):
                    or_part = or_part.strip()
                    negated = False
                    if or_part.startswith("~"):
                        or_part = or_part[1:].strip()
                        negated = True
                    if or_part.startswith("^"):
                        or_part = or_part[1:].strip()
                        condition = Q(text_prefix__istartswith=or_part)
                    else:
                        condition = Q(text__icontains=or_part.strip())
                    if negated:
                        condition = ~condition
                    if or_condition is None:
                        or_condition = condition
                    else:
                        or_condition |= condition
                if and_condition is None:
                    and_condition = or_condition
                else:
                    and_condition &= or_condition
            self = self.filter(and_condition)
        return self

    def tags(self, **params):
        params = TagParams(**params).model_dump()
        if params.get("any"):
            self = self.filter(example_tags__tags__overlap=params["any"])
        if params.get("all"):
            self = self.filter(example_tags__tags__contains=params["all"])
        if params.get("not_any"):
            self = self.exclude(example_tags__tags__overlap=params["not_any"])
        if params.get("not_all"):
            self = self.exclude(example_tags__tags__contains=params["not_all"])
        return self

    def semantic_sort(self, value):
        """Apply a semantic sort to the query."""
        if isinstance(value, str):
            value = batch_embed([value], dimensions=96)[0]
        assert isinstance(value, list), (
            "Semantic sort must be a string or list"
        )
        assert len(value) == 96, "Semantic sort must be a list of 96 floats"
        assert all(isinstance(v, float) for v in value), (
            "Semantic sort must be a list of floats"
        )
        return self.annotate(
            distance=CosineDistance("embedding", value)
        ).order_by("distance")

    def search(self, **query):
        """Search with params, pagination, and sorting."""
        project = self.model.get_project()

        # Prepare query
        if query.get("params", {}).get("tags"):
            query["params"]["tags"] = {
                k: get_tag_ids(v, project.id)
                for k, v in query["params"]["tags"].items()
                if v
            }

        # Validate query
        query = SearchQuery(**query).model_dump()
        self = self.annotate(tags=models.F("example_tags__tags"))

        active_label_id = query.get("active_label_id")
        label = (
            project.labels.get(id=active_label_id) if active_label_id else None
        )

        # Apply heuristic bucket filter
        heuristic_bucket = query["params"].get("heuristic_bucket")
        if label is not None and heuristic_bucket:
            if heuristic_bucket == "excluded":
                self = label.excluded_query(self)
            elif heuristic_bucket == "neutral":
                self = label.neutral_query(self)
            elif heuristic_bucket == "likely":
                self = label.likely_query(self)

        # Apply trainset split filter
        trainset_split = query["params"].get("trainset_split")
        if label is not None and trainset_split:
            if trainset_split == "train":
                self = self.tags(any=[label.trainset_train_tag.id])
            elif trainset_split == "eval":
                self = self.tags(any=[label.trainset_eval_tag.id])
            elif trainset_split == "both":
                self = self.tags(
                    any=[
                        label.trainset_train_tag.id,
                        label.trainset_eval_tag.id,
                    ]
                )

        # Apply predictor value filter
        predictor_value = query["params"].get("predictor_value")
        if label is not None and predictor_value:
            self = self.tags(
                any=[label.trainset_train_tag.id, label.trainset_eval_tag.id]
            )
            if predictor_value == "true":
                self = self.tags(any=[label.trainset_pred_tag.id])
            elif predictor_value == "false":
                self = self.tags(not_any=[label.trainset_pred_tag.id])

        # Apply manual annotation filter
        annotation_value = query["params"].get("annotation_value")
        if label is not None and annotation_value:
            if annotation_value == "true":
                self = self.tags(any=[label.anno_true_tag.id])
            elif annotation_value == "false":
                self = self.tags(any=[label.anno_false_tag.id])
            elif annotation_value == "flag":
                self = self.tags(any=[label.anno_flag_tag.id])
            elif annotation_value == "any":
                self = self.tags(
                    any=[
                        label.anno_true_tag.id,
                        label.anno_false_tag.id,
                        label.anno_flag_tag.id,
                    ]
                )
            elif annotation_value == "none":
                self = self.tags(
                    not_any=[
                        label.anno_true_tag.id,
                        label.anno_false_tag.id,
                        label.anno_flag_tag.id,
                    ]
                )

        # Apply param filters
        params = query["params"]
        self = self.tags(**params.get("tags", {}))
        self = self.querystring(params.get("querystring"))

        # Return count if requested
        if query.get("count"):
            return {"total": self.count()}

        # Apply sorting
        if query.get("semantic_sort"):
            self = self.semantic_sort(query["semantic_sort"])
        else:
            self = self.order_by(*query["sort"])

        # Select columns
        cols = ["id", "text_hash", "text", "tags"]
        self = self.values(*cols)

        # Apply pagination
        self = self.page(query["page"], size=query["page_size"])
        data = list(self)
        if label is not None and len(data):
            data = pd.DataFrame(data)
            trainset_examples = label.trainset_examples.filter(
                text_hash__in=data["text_hash"].tolist()
            )
            trainset_examples = trainset_examples.values(
                "text_hash", "split", "pred", "reason"
            )
            trainset_examples = pd.DataFrame(trainset_examples)
            if len(trainset_examples):
                trainset_examples = trainset_examples.drop_duplicates(
                    subset="text_hash"
                )
                data = data.merge(
                    trainset_examples, on="text_hash", how="left"
                )
            data = data.to_dict(orient="records")
        data = json.loads(json.dumps(data, ignore_nan=True))
        return {"data": data}

    def page(self, page, size=100):
        assert isinstance(page, int), "Page number must be an integer"
        assert page > 0, "Page number must be greater than 0"
        assert size > 0, "Page size must be greater than 0"
        assert size <= 1000, "Page size must be less than 1000"
        return self[size * (page - 1) : size * page]


# Queryset Managers
class SearchManager(CopyManager.from_queryset(SearchQuerySet)):
    pass


# Abstract Models
class BaseModel(models.Model):
    """Base model for all models"""

    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class SearchDocumentModelBase(models.base.ModelBase):
    """Meta class for search document models."""

    def __new__(cls, name, bases, attrs, **kwargs):
        """Create a new search document model."""
        if "Meta" not in attrs:
            project_id = attrs.get("project_id")
            if project_id is None:
                raise ValueError(f"{name} must define a project_id")
            attrs["Meta"] = type(
                "Meta",
                (),
                {
                    "db_table": f"project_{project_id}_doc",
                    "indexes": [
                        models.Index(
                            fields=["shuffle_sort", "id"],
                            name=f"{project_id}_s_idx",
                        ),
                        models.Index(
                            fields=["text_prefix"],
                            name=f"{project_id}_pr_idx",
                            opclasses=["text_pattern_ops"],
                        ),
                        GinIndex(
                            fields=["text"],
                            name=f"{project_id}_trg_idx",
                            opclasses=["gin_trgm_ops"],
                        ),
                        HnswIndex(
                            fields=["embedding"],
                            name=f"{project_id}_hnsw_idx",
                            m=16,
                            ef_construction=64,
                            opclasses=["halfvec_cosine_ops"],
                        ),
                    ],
                },
            )
        return super().__new__(cls, name, bases, attrs, **kwargs)


class SearchDocumentModel(BaseModel, metaclass=SearchDocumentModelBase):
    """Search document model."""

    project_id = None

    id = models.BigIntegerField(primary_key=True)
    text = models.TextField()
    text_prefix = models.CharField(max_length=50)
    text_hash = models.CharField(max_length=255)
    shuffle_sort = models.IntegerField()
    embedding = HalfVectorField(dimensions=96)

    objects = SearchManager()

    def save(self, *args, **kwargs):
        self.text_prefix = self.text[:50]
        self.text_hash = generate_hash(self.text)
        super().save(*args, **kwargs)

    @classmethod
    def get_project(cls):
        """Get the project for the search document."""
        return get_search_model_project(cls)

    @property
    def project(self):
        """Get the project for the search document."""
        return self.get_project()

    @classmethod
    def create_tags_model(cls):
        model_name = f"{cls.__name__}Tags"

        attrs = {
            "__module__": cls.__module__,
            "is_tag_model": True,
            "project_id": cls.project_id,
            "id": models.OneToOneField(
                cls,
                on_delete=models.CASCADE,
                primary_key=True,
                db_column="id",
                related_name="example_tags",
            ),
            "tags": ArrayField(
                models.BigIntegerField(), default=list, blank=True
            ),
            "objects": CopyManager(),
            "get_project": classmethod(get_search_model_project),
            "project": property(lambda self: self.get_project()),
            "Meta": type(
                "Meta",
                (),
                {
                    "db_table": f"project_{cls.project_id}_tags",
                    "indexes": [
                        GinIndex(
                            fields=["tags"],
                            name=f"{cls.project_id}_t_gin",
                        ),
                    ],
                },
            ),
        }

        TagsModel = type(model_name, (models.Model,), attrs)
        return TagsModel

    @classmethod
    def guarantee_tags_rows(cls):
        q = cls.objects.filter(example_tags__isnull=True)
        if q.exists():
            for data in q.batch_df("id", batch_size=500000):
                tags_model = cls.get_project().get_tags_model()
                f = StringIO()
                data.to_csv(f, index=False)
                f.seek(0)
                tags_model.objects.from_csv(
                    f,
                    static_mapping={"tags": "{}"},
                    ignore_conflicts=True,
                )

    @classmethod
    def bulk_replace_tag(cls, tag, ids):
        """Bulk replace a tags for a table."""
        tag_id = get_tag_ids([tag], cls.project_id)[0]
        tags_table = cls.get_project().get_tags_model()._meta.db_table
        added = removed = 0

        with transaction.atomic(), connection.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE stage_tag_ids(example_id BIGINT) ON COMMIT DROP;"
            )
            cur.execute("CREATE INDEX ON stage_tag_ids(example_id);")

            f = StringIO("".join(f"{i}\n" for i in ids))
            cur.copy_expert(
                "COPY stage_tag_ids (example_id) FROM STDIN WITH (FORMAT CSV)",
                f,
            )

            cur.execute(
                f"""
                UPDATE "{tags_table}" t
                SET tags = array_cat(t.tags, ARRAY[%s]::bigint[])
                FROM stage_tag_ids s
                WHERE t.id = s.example_id
                    AND NOT (t.tags @> ARRAY[%s]::bigint[])
                """,
                [tag_id, tag_id],
            )
            added = cur.rowcount

            cur.execute(
                f"""
                UPDATE "{tags_table}" t
                SET tags = array_remove(t.tags, %s)
                WHERE t.tags @> ARRAY[%s]::bigint[]
                    AND NOT EXISTS (
                        SELECT 1 FROM stage_tag_ids s WHERE s.example_id = t.id
                    )
                """,
                [tag_id, tag_id],
            )
            removed = cur.rowcount

        return added, removed

    @classmethod
    def bulk_update_column(cls, column, ids, values, id_column="id"):
        """Bulk update column values by ID."""
        assert len(ids) == len(values), "ids and values must match in length"

        field = cls._meta.get_field(column)
        field_type = get_pg_type(field)
        id_type = get_pg_type(cls._meta.get_field(id_column))

        table = cls._meta.db_table

        f = StringIO()
        writer = csv.writer(f)
        for k, v in zip(ids, values):
            writer.writerow([k, "" if v is None else v])
        f.seek(0)
        with transaction.atomic(), connection.cursor() as cur:
            cur.execute(
                f"CREATE TEMP TABLE stage_updates(id {id_type}, val text) ON COMMIT DROP;"
            )
            cur.copy_expert(
                "COPY stage_updates (id, val) FROM STDIN WITH (FORMAT CSV)",
                f,
            )
            cur.execute(
                f"""
                UPDATE "{table}" t
                SET {column} =
                    CASE WHEN s.val = '' THEN NULL ELSE s.val::{field_type} END
                FROM stage_updates s
                WHERE t.{id_column} = s.id
                """
            )
            updated = cur.rowcount
            return updated

    @classmethod
    def bulk_insert(cls, data, **kwargs):
        """Bulk insert data into the model."""
        if "id" not in data.columns:
            start_id = 1
            if cls.objects.exists():
                start_id = cls.objects.order_by("-id").first().id + 1
            data["id"] = range(start_id, start_id + len(data))
        data = data.dropna(subset=["text"])
        data["text"] = data["text"].str.strip()
        data = data[data["text"].apply(len) > 0]
        data["text_prefix"] = data["text"].apply(lambda x: x[:50])
        data["text_hash"] = data["text"].apply(generate_hash)
        data["shuffle_sort"] = data["text_hash"].apply(
            lambda x: random.randint(0, 100000000)
        )
        embeddings = data.copy().drop_duplicates(subset=["text_hash"])[
            ["text_hash", "text"]
        ]
        embeddings = cls.get_project().load_or_add_embeddings(embeddings)[
            ["text_hash", "embedding"]
        ]
        data = data.merge(embeddings, on="text_hash", how="left")
        f = StringIO()
        data.to_csv(f, index=False)
        f.seek(0)
        cls.objects.from_csv(
            f,
            static_mapping={
                "created_at": timezone.now(),
                "updated_at": timezone.now(),
            },
            **kwargs,
        )
        cls.guarantee_tags_rows()

    def set_annotation(self, label, value):
        """Set annotation tag for this example for the given label."""
        if isinstance(value, bool):
            value = "true" if value else "false"
        assert value is None or value in ["true", "false", "flag"], (
            "value must be 'true', 'false', 'flag', True, False, or None"
        )

        tags = self.example_tags
        tag_ids = {
            "true": label.anno_true_tag.id,
            "false": label.anno_false_tag.id,
            "flag": label.anno_flag_tag.id,
        }
        for tag_id in tag_ids.values():
            if tag_id in tags.tags:
                tags.tags.remove(tag_id)
        if value in tag_ids:
            tags.tags.append(tag_ids[value])
        tags.save()
        label.update_trainset_pred_counts()

    class Meta:
        abstract = True
        indexes = []


# Utils
def get_search_model_project(cls):
    """Get the project for a search document or search tags model."""
    if cls.project_id is not None:
        from .models import Project

        return Project.objects.get(id=cls.project_id)


def get_pg_type(field):
    """Get the PostgreSQL type for a field."""
    if isinstance(field, (models.IntegerField | models.BigIntegerField)):
        pg_type = "bigint"
    elif isinstance(field, (models.TextField | models.CharField)):
        pg_type = "text"
    else:
        raise NotImplementedError(f"Unsupported field type: {type(field)}")
    return pg_type


def get_tag_ids(tags, project_id):
    from clx.models import LabelTag

    if all(isinstance(tag, int) for tag in tags):
        return tags
    elif all(isinstance(tag, LabelTag) for tag in tags):
        return [tag.id for tag in tags]
    elif all(isinstance(tag, str) for tag in tags):
        tags = LabelTag.objects.filter(
            slug__in=tags, label__project__id=project_id
        )
        return [tag.id for tag in tags]
    else:
        raise ValueError(
            "tags must be same type, either int, LabelTag, or tag slug string"
        )


def init_search_models(**kwargs):
    """Create a project for each SearchDocumentModel.

    Then register the associated tags model.
    """
    from .models import Project

    for model in apps.get_models():
        if issubclass(model, SearchDocumentModel):
            project, created = Project.objects.get_or_create(
                id=model.project_id,
                model_name=model.__name__,
            )
            if created:
                project.name = model.__name__
                project.save()

    for model in apps.get_models():
        if hasattr(model, "is_tag_model") and model.is_tag_model:
            project = Project.objects.get(id=model.project_id)
            if project.tags_model_name != model.__name__:
                project.tags_model_name = model.__name__
                project.save()
