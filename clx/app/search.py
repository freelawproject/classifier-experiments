from __future__ import annotations

from typing import Literal

from django.db.models import F as models_F
from django.db.models import OuterRef, Q, Subquery
from django.db.models.fields import Field as DjangoField
from django.db.models.lookups import Lookup
from postgres_copy import CopyManager, CopyQuerySet
from pydantic import BaseModel

# ── Custom ILIKE lookup (uses trigram GIN index) ────────────


class ILike(Lookup):
    """Case-insensitive LIKE using ILIKE — supported by pg_trgm GIN index."""

    lookup_name = "ilike"

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        return f"{lhs} ILIKE {rhs}", [*lhs_params, *rhs_params]


DjangoField.register_lookup(ILike)

# ── Query Schema ─────────────────────────────────────────────


class Contains(BaseModel):
    type: Literal["contains"] = "contains"
    value: str


class StartsWith(BaseModel):
    type: Literal["startsWith"] = "startsWith"
    value: str


class Not(BaseModel):
    type: Literal["not"] = "not"
    query: Query


class Or(BaseModel):
    type: Literal["or"] = "or"
    queries: list[Query]


class And(BaseModel):
    type: Literal["and"] = "and"
    queries: list[Query]


Query = Contains | StartsWith | Not | Or | And

Not.model_rebuild()
Or.model_rebuild()
And.model_rebuild()


# ── Query String Parser ──────────────────────────────────────


def parse_query(text: str) -> dict:
    """Parse a shorthand query string into a query dict.

    Syntax:
        ,  = AND
        |  = OR
        ~  = NOT
        ^  = STARTSWITH
        () = grouping
        AND binds tighter than OR, so `A, B | C` = `A AND (B OR C)`
        ...wait, the user said the opposite: "ands scoped outside ors"
        meaning `A, B | C` = `A AND (B OR C)`.
        Actually re-reading: "commas=AND ... ands as scoped outside of ors,
        so A, B | C means A and (B or C)".
        This means AND has *lower* precedence than OR.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"type": "contains", "value": ""}
    parser = _Parser(tokens)
    result = parser.parse_expr()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected token: {parser.peek()}")
    return result


def _tokenize(text: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ",|~^()":
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in ",|~^()":
                j += 1
            tokens.append(text[i:j].strip())
            i = j
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"Unexpected end of input (expected {expected})")
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse_expr(self) -> dict:
        return self.parse_and()

    def parse_and(self) -> dict:
        parts = [self.parse_or()]
        while self.peek() == ",":
            self.consume(",")
            parts.append(self.parse_or())
        return (
            parts[0] if len(parts) == 1 else {"type": "and", "queries": parts}
        )

    def parse_or(self) -> dict:
        parts = [self.parse_unary()]
        while self.peek() == "|":
            self.consume("|")
            parts.append(self.parse_unary())
        return (
            parts[0] if len(parts) == 1 else {"type": "or", "queries": parts}
        )

    def parse_unary(self) -> dict:
        if self.peek() == "~":
            self.consume("~")
            return {"type": "not", "query": self.parse_unary()}
        return self.parse_primary()

    def parse_primary(self) -> dict:
        if self.peek() == "(":
            self.consume("(")
            expr = self.parse_expr()
            self.consume(")")
            return expr
        if self.peek() == "^":
            self.consume("^")
            return {"type": "startsWith", "value": self.consume()}
        return {"type": "contains", "value": self.consume()}


# ── Django Q Builder ─────────────────────────────────────────


def build_q(query: dict) -> Q:
    """Convert a query dict into a Django Q object.

    Uses ILIKE for contains (trigram GIN index compatible).
    startsWith uses text_prefix for better index utilization.
    """
    match query["type"]:
        case "contains":
            return Q(text__ilike=f"%{query['value']}%")
        case "startsWith":
            return Q(text_prefix__ilike=f"{query['value']}%")
        case "not":
            return ~build_q(query["query"])
        case "or":
            result = Q()
            for sub in query["queries"]:
                result |= build_q(sub)
            return result
        case "and":
            result = Q()
            for sub in query["queries"]:
                result &= build_q(sub)
            return result
        case _:
            raise ValueError(f"Unknown query type: {query['type']}")


# ── Custom QuerySet & Manager ────────────────────────────────


class SearchQuerySet(CopyQuerySet):
    def text_query(self, query: dict) -> SearchQuerySet:
        """Filter using a query dict (matching the Query schema)."""
        return self.filter(build_q(query))

    def query_string(self, qs: str) -> SearchQuerySet:
        """Filter using the shorthand query string syntax."""
        return self.text_query(parse_query(qs))

    def training_examples(self, label_id: str) -> SearchQuerySet:
        """Filter to documents in a label's training set."""
        return self.filter(label_documents__label_id=label_id)

    def with_annotation(
        self, label_id: str, source: str = "agent"
    ) -> SearchQuerySet:
        """Annotate each document with its classification value for label+source.

        Adds an `annotation_value` field (str or None) to each row.
        Uses a single subquery — no N+1.
        """
        from clx.app.models import ClassificationAnnotation

        return self.annotate(
            annotation_value=Subquery(
                ClassificationAnnotation.objects.filter(
                    label_document__document=OuterRef("pk"),
                    label_document__label_id=label_id,
                    source=source,
                ).values("value")[:1]
            )
        )

    def filter_annotation(
        self, label_id: str, value: str, source: str = "agent"
    ) -> SearchQuerySet:
        """Filter documents by annotation value for a label+source.

        value: 'yes', 'no', 'skip', 'none' (unannotated), or 'any' (has annotation).
        """
        qs = self.training_examples(label_id).with_annotation(label_id, source)
        if value == "none":
            return qs.filter(annotation_value__isnull=True)
        if value == "any":
            return qs.filter(annotation_value__isnull=False)
        return qs.filter(annotation_value=value)

    def with_prediction(self, label_id: str) -> SearchQuerySet:
        """Annotate each document with its prediction value and confidence.

        Adds `prediction_value` (str or None) and `prediction_confidence`
        (float or None) fields to each row.
        """
        from clx.app.models import LabelDocument

        return self.annotate(
            prediction_value=Subquery(
                LabelDocument.objects.filter(
                    document=OuterRef("pk"),
                    label_id=label_id,
                ).values("prediction")[:1]
            ),
            prediction_confidence_value=Subquery(
                LabelDocument.objects.filter(
                    document=OuterRef("pk"),
                    label_id=label_id,
                ).values("prediction_confidence")[:1]
            ),
        )

    def filter_prediction(self, label_id: str, value: str) -> SearchQuerySet:
        """Filter documents by prediction value for a label.

        value: 'yes', 'no', 'any', or 'disagree' (prediction != agent annotation).
        """
        from clx.app.models import LabelDocument

        qs = self.training_examples(label_id)
        qs = qs.annotate(
            prediction_value=Subquery(
                LabelDocument.objects.filter(
                    document=OuterRef("pk"),
                    label_id=label_id,
                ).values("prediction")[:1]
            )
        )
        if value == "disagree":
            qs = qs.with_annotation(label_id, "agent")
            return qs.filter(
                prediction_value__isnull=False,
                annotation_value__isnull=False,
            ).exclude(prediction_value=models_F("annotation_value"))
        if value == "any":
            return qs.exclude(prediction_value="").filter(
                prediction_value__isnull=False
            )
        return qs.filter(prediction_value=value)


class SearchManager(CopyManager.from_queryset(SearchQuerySet)):
    pass
