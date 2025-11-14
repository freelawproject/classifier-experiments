import csv
import random
from io import StringIO

import pandas as pd
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import connection, models, transaction
from django.db.models import Q
from django.utils import timezone
from postgres_copy import CopyManager, CopyQuerySet
from pydantic import BaseModel as PydanticModel

from clx.utils import generate_hash


# Pydantic Models
class FeatureQuery(PydanticModel):
    any: list[int] = []
    all: list[int] = []
    not_any: list[int] = []
    not_all: list[int] = []


class SearchQuery(PydanticModel):
    features: FeatureQuery = FeatureQuery()
    querystring: str | None = None
    sort: str = "shuffle_sort"


class EndpointQuery(SearchQuery):
    page: int = 1
    page_size: int = 100


# QuerySets
class SearchQuerySet(CopyQuerySet):
    """QuerySet for search queries."""

    def querystring(self, query):
        """Apply a querystring to the query.

        For querystrings we will do an exact substring match on terms / phrases.
        Commas will be treated as AND operators.
        Bars will be treated as OR operators.
        Tildes will be treated as NOT operators.
        Carets will be treated as startswith operators.
        We will always assume that ORs are nested in ANDs.
        """
        assert isinstance(query, str), "Querystring must be a string"
        and_condition = None
        for and_part in query.split(","):
            or_condition = None
            for or_part in and_part.split("|"):
                or_part = or_part.strip()
                if or_part.startswith("~"):
                    or_part = or_part[1:]
                    condition = ~Q(text__icontains=or_part.strip())
                elif or_part.startswith("^"):
                    or_part = or_part[1:]
                    condition = Q(text_prefix__istartswith=or_part.strip())
                else:
                    condition = Q(text__icontains=or_part)
                if or_condition is None:
                    or_condition = condition
                else:
                    or_condition |= condition
            if and_condition is None:
                and_condition = or_condition
            else:
                and_condition &= or_condition
        return self.filter(and_condition)

    def features(self, **query):
        query = {
            k: get_feature_ids(v, self.model_name)
            for k, v in query.items()
            if v
        }
        query = FeatureQuery(**query).model_dump()
        if query.get("any"):
            self = self.filter(features__overlap=query["any"])
        if query.get("all"):
            self = self.filter(features__contains=query["all"])
        if query.get("not_any"):
            self = self.exclude(features__overlap=query["not_any"])
        if query.get("not_all"):
            self = self.exclude(features__contains=query["not_all"])
        return self

    def search(self, **query):
        """Search for docket entries by query."""
        if "features" in query:
            query["features"] = {
                k: get_feature_ids(v, self.model_name)
                for k, v in query["features"].items()
                if v
            }
        query = SearchQuery(**query).model_dump()
        if query.get("querystring"):
            self = self.querystring(query["querystring"])
        if query.get("features"):
            self = self.features(**query["features"])
        self = self.order_by(query["sort"])
        return self

    def page(self, page, size=100):
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        elif page > 100:
            raise ValueError("Page number must be less than 100")
        if size < 1:
            raise ValueError("Page size must be greater than 0")
        elif size > 1000:
            raise ValueError("Page size must be less than 1000")
        return self[size * (page - 1) : size * page]

    def _chain(self):
        clone = super()._chain()
        clone.model_name = self.model_name
        return clone


# Queryset Managers
class SearchManager(CopyManager.from_queryset(SearchQuerySet)):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = None

    def get_queryset(self):
        queryset = super().get_queryset()
        queryset.model_name = self.model_name
        return queryset


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
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)
        if not new_cls._meta.abstract:
            new_cls.objects.model_name = new_cls.__name__
            new_cls._meta.indexes = [
                GinIndex(
                    fields=["features"],
                    name=f"{new_cls._meta.db_table}_fs_gin",
                ),
                models.Index(
                    fields=["shuffle_sort"],
                    name=f"{new_cls._meta.db_table}_ss_idx",
                ),
                models.Index(
                    fields=["id"],
                    name=f"{new_cls._meta.db_table}_id_idx",
                ),
                models.Index(
                    fields=["text_prefix"],
                    name=f"{new_cls._meta.db_table}_pr_idx",
                    opclasses=["text_pattern_ops"],
                ),
                GinIndex(
                    fields=["text"],
                    name=f"{new_cls._meta.db_table}_trgm_idx",
                    opclasses=["gin_trgm_ops"],
                ),
            ]
        return new_cls


class SearchDocumentModel(BaseModel, metaclass=SearchDocumentModelBase):
    """Search document model."""

    id = models.BigIntegerField(primary_key=True)
    text = models.TextField()
    text_prefix = models.CharField(max_length=50)
    text_hash = models.CharField(max_length=255)
    features = ArrayField(models.BigIntegerField(), default=list, blank=True)
    shuffle_sort = models.IntegerField()

    objects = SearchManager()

    def save(self, *args, **kwargs):
        self.text_prefix = self.text[:50]
        self.text_hash = generate_hash(self.text)
        super().save(*args, **kwargs)

    @classmethod
    def batch_df(cls, *columns, batch_size=1000000):
        last_id = None
        while 1:
            q = cls.objects.all()
            if last_id is not None:
                q = q.filter(id__gt=last_id)
            q = q.order_by("id").values(*columns)[:batch_size]
            data = pd.DataFrame(q)
            if len(data) == 0:
                break
            yield data
            last_id = data["id"].max()

    @classmethod
    def get_project(cls):
        """Get the project for the search document."""
        if not cls._meta.abstract:
            from clx.models import Project

            project, _ = Project.objects.get_or_create(
                name=cls.__name__,
                model_name=cls.__name__,
                slug=cls.__name__.lower(),
            )
            return project

    @property
    def project(self):
        """Get the project for the search document."""
        return self.get_project()

    @classmethod
    def bulk_replace_feature(cls, feature_id, ids):
        """Bulk replace a feature for a table."""
        entry_table = cls._meta.db_table
        added = removed = 0

        with transaction.atomic(), connection.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE stage_feature_ids(entry_id BIGINT) ON COMMIT DROP;"
            )
            cur.execute("CREATE INDEX ON stage_feature_ids(entry_id);")

            buf = StringIO("".join(f"{i}\n" for i in ids))
            cur.copy_expert(
                "COPY stage_feature_ids (entry_id) FROM STDIN WITH (FORMAT CSV)",
                buf,
            )

            cur.execute(
                f"""
                UPDATE {entry_table} e
                SET features = array_cat(e.features, ARRAY[%s]::bigint[])
                FROM stage_feature_ids s
                WHERE e.id = s.entry_id
                    AND NOT (ARRAY[%s]::bigint[] <@ e.features)
                """,
                [feature_id, feature_id],
            )
            added = cur.rowcount

            cur.execute(
                f"""
                UPDATE {entry_table} e
                SET features = array_remove(e.features, %s)
                WHERE (ARRAY[%s]::bigint[] <@ e.features)
                    AND NOT EXISTS (
                        SELECT 1 FROM stage_feature_ids s WHERE s.entry_id = e.id
                    )
                """,
                [feature_id, feature_id],
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

        buf = StringIO()
        writer = csv.writer(buf)
        for k, v in zip(ids, values):
            writer.writerow([k, "" if v is None else v])
        buf.seek(0)
        with transaction.atomic(), connection.cursor() as cur:
            cur.execute(
                f"CREATE TEMP TABLE stage_updates(id {id_type}, val text) ON COMMIT DROP;"
            )
            cur.copy_expert(
                "COPY stage_updates (id, val) FROM STDIN WITH (FORMAT CSV)",
                buf,
            )
            cur.execute(
                f"""
                UPDATE {table} t
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
        data["text_prefix"] = data["text"].apply(lambda x: x[:50])
        data["text_hash"] = data["text"].apply(generate_hash)
        data["shuffle_sort"] = data["text_hash"].apply(
            lambda x: random.randint(0, 100000000)
        )
        f = StringIO()
        data.to_csv(f, index=False)
        f.seek(0)
        cls.objects.from_csv(
            f,
            static_mapping={
                "created_at": timezone.now(),
                "updated_at": timezone.now(),
                "features": "{}",
            },
            **kwargs,
        )

    class Meta:
        abstract = True
        ordering = ["shuffle_sort"]
        indexes = []


# Utils
def get_pg_type(field):
    """Get the PostgreSQL type for a field."""
    if isinstance(field, (models.IntegerField | models.BigIntegerField)):
        pg_type = "bigint"
    elif isinstance(field, (models.TextField | models.CharField)):
        pg_type = "text"
    else:
        raise NotImplementedError(f"Unsupported field type: {type(field)}")
    return pg_type


def get_feature_ids(features, model_name):
    from clx.models import LabelFeature

    if all(isinstance(f, int) for f in features):
        return features
    elif all(isinstance(f, LabelFeature) for f in features):
        return [f.id for f in features]
    elif all(isinstance(f, str) for f in features):
        features = LabelFeature.objects.filter(
            slug__in=features, label__project__model_name=model_name
        )
        return [f.id for f in features]
    else:
        raise ValueError(
            "features must be same type, either int, LabelFeature, or feature slug string"
        )
