from django.db.models import Q
from postgres_copy import CopyManager, CopyQuerySet
from pydantic import BaseModel as PydanticModel


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
                    condition = Q(text__istartswith=or_part.strip())
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
        query = SearchQuery(**query).model_dump()
        if query.get("querystring"):
            self = self.querystring(query["querystring"])
        if query.get("features"):
            self = self.features(**query["features"])
        self = self.order_by(query["sort"])
        return self

    def page(self, page, page_size=100):
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        elif page > 100:
            raise ValueError("Page number must be less than 100")
        if page_size < 1:
            raise ValueError("Page size must be greater than 0")
        elif page_size > 1000:
            raise ValueError("Page size must be less than 1000")
        return self[page_size * (page - 1) : page_size * page]

    def _chain(self):
        clone = super()._chain()
        clone.table_name = self.table_name
        return clone


# Queryset Managers
class SearchManager(CopyManager.from_queryset(SearchQuerySet)):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_name = None

    def get_queryset(self):
        queryset = super().get_queryset()
        queryset.table_name = self.table_name
        return queryset
