import logging
import time
from typing import Literal

from pydantic import BaseModel, Field
from shortuuid import ShortUUID

from clx.app.search import Query
from clx.llm.agent import Tool

_su = ShortUUID()
logger = logging.getLogger("clx.llm")


class Search(Tool):
    """Search documents in the project using a structured query.

    Build queries using Contains, StartsWith, Not, Or, and And nodes.
    All text matching is case-insensitive.

    Performance tip: complex queries with many OR branches are expensive.
    Prefer running several simple searches over one large compound query.
    That said, don't shy away from And/Not/Or when you genuinely need
    the expressive power — just keep each individual query focused.
    """

    query: Query | None = Field(
        default=None,
        description=(
            "A structured query object. Use {type:'contains', value:'...'} "
            "for text search, {type:'startsWith', value:'...'} for prefix "
            "search, {type:'not', query:...} for negation, "
            "{type:'or', queries:[...]} for OR, "
            "{type:'and', queries:[...]} for AND. "
            "Omit to match all documents."
        ),
    )
    num_results: int = Field(
        default=10, description="Number of results to return (max 1000)"
    )
    offset: int = Field(
        default=0,
        description="Number of results to skip (for pagination).",
    )
    sort: Literal["shuffle", "tricky"] = Field(
        default="shuffle",
        description=(
            "Sort order: 'shuffle' (default random order) or 'tricky' "
            "(ascending prediction confidence — least confident first)."
        ),
    )

    from_training_set: bool = Field(
        default=False,
        description=(
            "If true, only search documents in the current "
            "label's training set."
        ),
    )
    annotation: str | None = Field(
        default=None,
        description=(
            "Filter by annotation value: 'yes', 'no', 'skip', "
            "'none' (unannotated), or 'any' (has any annotation). "
            "Implies from_training_set."
        ),
    )
    prediction: str | None = Field(
        default=None,
        description=(
            "Filter by prediction value: 'yes', 'no', 'any' "
            "(has any prediction), or 'disagree' (prediction differs "
            "from agent annotation). Implies from_training_set."
        ),
    )
    count_only: bool = Field(
        default=False,
        description=(
            "If true, return only the count of matching documents "
            "(num_results is ignored)."
        ),
    )

    def __call__(self, agent):
        from django.db.models import F as models_F
        from django.db.models import OuterRef, Subquery

        from clx.app.models import LabelDocument

        project = agent.thread.label.project
        label_id = agent.thread.label_id

        if self.sort == "tricky":
            documents = project.documents.annotate(
                _pred_conf=Subquery(
                    LabelDocument.objects.filter(
                        document=OuterRef("pk"),
                        label_id=label_id,
                    ).values("prediction_confidence")[:1]
                )
            ).order_by(models_F("_pred_conf").asc(nulls_last=True))
        else:
            documents = project.documents.order_by("shuffle_key")
        if self.query:
            documents = documents.text_query(self.query.model_dump())
        if self.prediction:
            documents = documents.filter_prediction(label_id, self.prediction)
        elif self.annotation:
            documents = documents.filter_annotation(label_id, self.annotation)
        elif self.from_training_set:
            documents = documents.training_examples(label_id)

        if self.count_only:
            t0 = time.time()
            total = documents.count()
            logger.debug(f"    Search count query: {time.time() - t0:.1f}s")
            return f"{total} document(s) match."

        num_results = min(self.num_results, 1000)
        t0 = time.time()
        rows = list(
            documents.values_list("id", "text")[
                self.offset : self.offset + num_results
            ]
        )
        db_elapsed = time.time() - t0
        query_desc = self.query.model_dump() if self.query else "all"
        logger.debug(
            f"    Search DB: {db_elapsed:.1f}s | "
            f"{len(rows)} rows | query={query_desc}"
        )

        # Generate a short search ID and store in agent state.
        search_id = _su.random(length=8)
        doc_ids = [str(r[0]) for r in rows]
        searches = agent.state.setdefault("searches", {})
        searches[search_id] = {
            "query": self.query.model_dump() if self.query else None,
            "num_results": num_results,
            "from_training_set": self.from_training_set,
            "result_count": len(rows),
            "document_ids": doc_ids,
        }

        if not rows:
            return f"[search:{search_id}] No documents found."
        lines = [f"doc_id={r[0]}\n{r[1]}" for r in rows]
        return (
            f"[search:{search_id}] {len(rows)} results:\n\n"
            + "\n---\n".join(lines)
        )


class UpdateLabelInstructions(Tool):
    """Update the instructions for the current label.

    Use 'append' to add new content to the end of existing instructions,
    or 'replace' to rewrite them entirely. You can see the current
    instructions in your system prompt.
    """

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(
        description="The instruction content to append or replace with"
    )

    def __call__(self, agent):
        label = agent.thread.label
        if self.mode == "append":
            if label.instructions.strip():
                label.instructions = (
                    label.instructions.rstrip() + "\n\n" + self.content
                )
            else:
                label.instructions = self.content
        else:
            label.instructions = self.content
        label.save(update_fields=["instructions", "updated_at"])
        return f"Label instructions updated ({self.mode})."


class UpdateProjectInstructions(Tool):
    """Update the project-level instructions.

    Use 'append' to add new content to the end of existing instructions,
    or 'replace' to rewrite them entirely. You can see the current
    instructions in your system prompt.
    """

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(
        description="The instruction content to append or replace with"
    )

    def __call__(self, agent):
        project = agent.thread.label.project
        if self.mode == "append":
            if project.instructions.strip():
                project.instructions = (
                    project.instructions.rstrip() + "\n\n" + self.content
                )
            else:
                project.instructions = self.content
        else:
            project.instructions = self.content
        project.save(update_fields=["instructions", "updated_at"])
        return f"Project instructions updated ({self.mode})."


class AddTrainingExamples(Tool):
    """Add documents to the current label's training set.

    These become reference examples for classification. Preferred: pass
    search_id from a previous Search call — this re-runs the query so you
    can sample more documents than were originally returned (use num_docs
    to control how many). Alternatively, pass explicit document_ids.
    Do not pass both.
    """

    search_id: str | None = Field(
        default=None,
        description=(
            "A search ID from a previous Search call (e.g. 'aBcDeFgH'). "
            "Re-runs the query and adds up to num_docs documents."
        ),
    )
    num_docs: int | None = Field(
        default=None,
        description=(
            "Max number of documents to add. Only used with search_id. "
            "Defaults to all matching documents."
        ),
    )
    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of document IDs "
            "(short UUIDs, NOT document text) to add."
        ),
    )

    def __call__(self, agent):
        from clx.app.models import Document, LabelDocument

        label = agent.thread.label

        if self.search_id and self.document_ids:
            return "Error: provide search_id or document_ids, not both."
        if not self.search_id and not self.document_ids:
            return "Error: provide either search_id or document_ids."

        if self.search_id:
            searches = agent.state.get("searches", {})
            search = searches.get(self.search_id)
            if not search:
                return f"Error: search '{self.search_id}' not found in state."
            # Re-run the query to get document IDs (not limited to original result size).
            project = label.project
            documents = project.documents.order_by("shuffle_key")
            query = search.get("query")
            if query:
                documents = documents.text_query(query)
            if search.get("from_training_set"):
                documents = documents.training_examples(label.id)
            if self.num_docs is not None:
                documents = documents[: self.num_docs]
            doc_ids = list(documents.values_list("id", flat=True))
        else:
            doc_ids = self.document_ids

        # Validate that all IDs actually exist to avoid FK violations.
        valid_ids = set(
            Document.objects.filter(
                id__in=doc_ids, project=label.project
            ).values_list("id", flat=True)
        )
        doc_ids = [did for did in doc_ids if did in valid_ids]

        if not doc_ids:
            return "Error: none of the provided document IDs are valid."

        objects = [
            LabelDocument(id=_su.uuid(), label=label, document_id=did)
            for did in doc_ids
        ]
        LabelDocument.objects.bulk_create(objects, ignore_conflicts=True)
        return f"Added {len(doc_ids)} training example(s) to label '{label.name}' (duplicates ignored)."


class AnnotationItem(BaseModel):
    document_id: str = Field(
        description="The document ID (short UUID from search results)."
    )
    value: Literal["yes", "no", "skip"] = Field(
        description="The classification value."
    )


class Annotate(Tool):
    """Annotate training examples for the current label.

    Each annotation classifies a document as 'yes', 'no', or 'skip'.
    If an annotation already exists for a document it will be updated.
    """

    annotations: list[AnnotationItem] = Field(
        description="List of annotations to create or update."
    )

    def __call__(self, agent):
        from clx.app.models import ClassificationAnnotation, LabelDocument

        label = agent.thread.label
        doc_ids = [a.document_id for a in self.annotations]

        # Fetch all LabelDocument rows for these docs+label in one query.
        ld_map = dict(
            LabelDocument.objects.filter(
                label=label, document_id__in=doc_ids
            ).values_list("document_id", "id")
        )

        missing = [did for did in doc_ids if did not in ld_map]
        # Deduplicate by document_id (last wins) to avoid
        # ON CONFLICT DO UPDATE hitting the same row twice.
        valid_by_id = {
            a.document_id: a
            for a in self.annotations
            if a.document_id in ld_map
        }
        valid = list(valid_by_id.values())

        # Bulk upsert annotations for valid docs.
        if valid:
            objects = [
                ClassificationAnnotation(
                    id=_su.uuid(),
                    label_document_id=ld_map[a.document_id],
                    value=a.value,
                    source="agent",
                )
                for a in valid
            ]
            ClassificationAnnotation.objects.bulk_create(
                objects,
                update_conflicts=True,
                unique_fields=["label_document", "source"],
                update_fields=["value", "updated_at"],
            )

        parts = [f"Annotated {len(valid)} document(s)."]
        if missing:
            parts.append(
                f"Skipped {len(missing)} not in training set (add with AddTrainingExamples): {', '.join(missing)}"
            )
        return " ".join(parts)


class CompactMemory(Tool):
    """Compact the conversation by replacing prior messages with a summary.

    ONLY call this tool when the user explicitly asks to compact or
    summarize the conversation. Write a detailed, verbose summary that
    captures all important context, decisions, findings, and state so
    nothing is lost. Try to keep the summary under 2500 tokens and avoid
    repeating content in your system prompt.
    """

    summary: str = Field(
        description=(
            "A detailed summary of the conversation so far. Be verbose — "
            "include key findings, decisions made, tool results, "
            "instructions given, and any state the agent should remember. "
            "This will replace all prior messages."
        )
    )

    def __call__(self, agent):
        return self.summary


class ClearToolHistory(Tool):
    """Clear tool calls and responses from the conversation history.

    This hides all previous tool call and tool response messages from
    the LLM context while keeping them in the database and UI. Use this
    to reduce context size when tool history is no longer needed.
    """

    def __call__(self, agent):
        from clx.app.models import Message as MessageModel

        # Find the current assistant message (contains this tool call).
        current_assistant_idx = None
        for i in range(len(agent.messages) - 1, -1, -1):
            msg = agent.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                current_assistant_idx = i
                break

        # Hide all tool-related messages before the current assistant message.
        for i, msg in enumerate(agent.messages):
            if i == current_assistant_idx:
                continue
            if msg.get("role") == "tool" or msg.get("tool_calls"):
                msg["hidden"] = True

        # Update DB — hide all existing tool-related messages.
        # Current step's messages haven't been saved yet, so this is safe.
        MessageModel.objects.filter(
            thread=agent.thread, hidden=False, data__role="tool"
        ).update(hidden=True)
        MessageModel.objects.filter(thread=agent.thread, hidden=False).filter(
            data__has_key="tool_calls"
        ).update(hidden=True)

        return "Tool history cleared from context."


class AskUser(Tool):
    """Ask the user a question with proposed answer options.

    The user will pick exactly one option, so make the options mutually
    exclusive. Use this when you need clarification or want the user to
    choose between alternatives. The question and options will be
    presented to the user in an interactive card.
    """

    question: str = Field(description="The question to ask the user")
    options: list[str] = Field(
        description=(
            "A list of mutually exclusive answer options (user picks one)"
        )
    )

    def __call__(self, agent):
        return "This question will be presented to the user."


class CompleteTask(Tool):
    """Mark the current task as complete.

    Call this when you have fully accomplished the task objective.
    """

    summary: str = Field(
        description="A brief summary of what was accomplished."
    )

    def __call__(self, agent):
        return self.summary
