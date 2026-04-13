import json

from django.conf import settings as django_settings
from django.db.models import Count, Max, OuterRef, Q, Subquery, Sum
from django.db.models import F as models_F
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import Label, Project, Prompt, Task, Thread

DOCS_PER_PAGE = 100

# --- Page Views ---


def projects_view(request):
    """Projects grid page."""
    projects = Project.objects.all().order_by("name")
    return render(request, "projects.html", {"projects": projects})


def project_detail_view(request, project_id):
    """Project detail page with paginated documents."""
    project = get_object_or_404(Project, id=project_id)
    # Lazily create an initial label if none exist.
    if not project.labels.exists():
        label = Label.objects.create(project=project, name="Initial Label")
        project.active_label = label
        project.save(update_fields=["active_label", "updated_at"])
    elif project.active_label is None:
        project.active_label = project.labels.first()
        project.save(update_fields=["active_label", "updated_at"])
    return render(
        request,
        "project_detail.html",
        {
            "project": project,
            "model_ids": django_settings.MODEL_IDS,
            "default_model": django_settings.DEFAULT_MODEL,
        },
    )


# --- API Endpoints ---


@csrf_exempt
@require_http_methods(["GET", "POST"])
def projects_api(request):
    """GET: list projects. POST: create a project."""
    if request.method == "GET":
        projects = Project.objects.all().order_by("name")
        return JsonResponse(
            {"projects": [{"id": str(p.id), "name": p.name} for p in projects]}
        )
    else:
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        name = data.get("name", "").strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        project = Project.objects.create(name=name)
        return JsonResponse(
            {"id": str(project.id), "name": project.name},
            status=201,
        )


def _filtered_documents(project, request):
    """Return a queryset of documents filtered by request params.

    Returns (queryset, label_id, has_annotation_col).
    """
    sort = request.GET.get("sort", "shuffle").strip()
    label_id = request.GET.get("label", "").strip()

    if sort == "tricky" and label_id:
        from .models import LabelDocument

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

    qs = request.GET.get("q", "").strip()
    if qs:
        documents = documents.query_string(qs)
    annotation = request.GET.get("annotation", "").strip()
    prediction = request.GET.get("prediction", "").strip()
    has_annotation_col = False
    if prediction and label_id:
        documents = documents.filter_prediction(label_id, prediction)
        if prediction == "disagree":
            has_annotation_col = True
    elif annotation and label_id:
        documents = documents.filter_annotation(label_id, annotation)
        has_annotation_col = True
    elif label_id:
        documents = documents.training_examples(label_id)
    return documents, label_id, has_annotation_col


@require_GET
def project_docs_api(request, project_id):
    """GET: paginated documents for a project."""
    project = get_object_or_404(Project, id=project_id)
    page_number = max(1, int(request.GET.get("page", 1)))
    documents, label_id, has_annotation_col = _filtered_documents(
        project, request
    )
    # Attach annotation and prediction values via subqueries when a label is active.
    if label_id and not has_annotation_col:
        documents = documents.with_annotation(label_id)
    if label_id:
        documents = documents.with_prediction(label_id)
    # Fetch one extra to detect if there's a next page, avoiding COUNT query.
    offset = (page_number - 1) * DOCS_PER_PAGE
    batch = list(documents[offset : offset + DOCS_PER_PAGE + 1])
    has_next = len(batch) > DOCS_PER_PAGE
    page_docs = batch[:DOCS_PER_PAGE]
    return JsonResponse(
        {
            "documents": [
                {
                    "id": str(d.id),
                    "text": d.text,
                    "text_prefix": d.text_prefix,
                    "meta": d.meta,
                    "annotation": getattr(d, "annotation_value", None),
                    "prediction": getattr(d, "prediction_value", None) or None,
                    "prediction_confidence": getattr(
                        d, "prediction_confidence_value", None
                    ),
                }
                for d in page_docs
            ],
            "page": page_number,
            "has_next": has_next,
        }
    )


@require_GET
def project_docs_count_api(request, project_id):
    """GET: count of documents for a project (with filters)."""
    project = get_object_or_404(Project, id=project_id)
    documents, _, _ = _filtered_documents(project, request)
    return JsonResponse({"total_count": documents.count()})


@require_GET
def project_labels_api(request, project_id):
    """GET: list labels for a project."""
    project = get_object_or_404(Project, id=project_id)
    labels = project.labels.order_by("name")

    # Aggregate stats for all labels in a single query.
    from .models import LabelDocument

    stats_qs = (
        LabelDocument.objects.filter(label__project=project)
        .values("label_id")
        .annotate(
            total=Count("id"),
            yes=Count(
                "id",
                filter=Q(
                    annotations__value="yes", annotations__source="agent"
                ),
            ),
            no=Count(
                "id",
                filter=Q(annotations__value="no", annotations__source="agent"),
            ),
            skip=Count(
                "id",
                filter=Q(
                    annotations__value="skip", annotations__source="agent"
                ),
            ),
            latest_annotation_at=Max(
                "annotations__updated_at",
                filter=Q(
                    annotations__source="agent",
                    annotations__value__in=["yes", "no"],
                ),
            ),
        )
    )
    stats_by_label = {}
    for row in stats_qs:
        lid = str(row["label_id"])
        total = row["total"]
        lat = row["latest_annotation_at"]
        stats_by_label[lid] = {
            "total": total,
            "yes": row["yes"],
            "no": row["no"],
            "skip": row["skip"],
            "unannotated": total - row["yes"] - row["no"] - row["skip"],
            "latest_annotation_at": lat.isoformat() if lat else None,
        }

    return JsonResponse(
        {
            "labels": [
                {
                    "id": str(label.id),
                    "name": label.name,
                    "instructions": label.instructions,
                    "finetune_status": label.finetune_status,
                    "finetuned_at": label.finetuned_at.isoformat()
                    if label.finetuned_at
                    else None,
                    "predicted_at": label.predicted_at.isoformat()
                    if label.predicted_at
                    else None,
                    "prediction_stats": label.prediction_stats,
                    "stats": stats_by_label.get(
                        str(label.id),
                        {
                            "total": 0,
                            "yes": 0,
                            "no": 0,
                            "skip": 0,
                            "unannotated": 0,
                        },
                    ),
                }
                for label in labels
            ],
            "active_label_id": str(project.active_label_id)
            if project.active_label_id
            else None,
        }
    )


@require_GET
def label_stats_api(request, project_id, label_id):
    """GET: annotation stats for a label's training set."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    from .models import LabelDocument

    stats = LabelDocument.objects.filter(label=label).aggregate(
        total=Count("id"),
        yes=Count(
            "id",
            filter=Q(annotations__value="yes", annotations__source="agent"),
        ),
        no=Count(
            "id",
            filter=Q(annotations__value="no", annotations__source="agent"),
        ),
        skip=Count(
            "id",
            filter=Q(annotations__value="skip", annotations__source="agent"),
        ),
    )
    stats["unannotated"] = (
        stats["total"] - stats["yes"] - stats["no"] - stats["skip"]
    )
    return JsonResponse(stats)


@csrf_exempt
@require_http_methods(["POST"])
def create_label_api(request, project_id):
    """POST: create a new label for a project."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    label = Label.objects.create(project=project, name=name)
    return JsonResponse({"id": str(label.id), "name": label.name}, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def set_active_label_api(request, project_id, label_id):
    """POST: set the active label for a project."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    project.active_label = label
    project.save(update_fields=["active_label", "updated_at"])
    return JsonResponse({"id": str(label.id), "name": label.name})


@csrf_exempt
@require_http_methods(["POST"])
def rename_label_api(request, project_id, label_id):
    """POST: update a label (name, instructions)."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "name" in data:
        name = data["name"].strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        label.name = name
        update_fields.append("name")
    if "instructions" in data:
        label.instructions = data["instructions"]
        update_fields.append("instructions")
    label.save(update_fields=update_fields)
    return JsonResponse(
        {
            "id": str(label.id),
            "name": label.name,
            "instructions": label.instructions,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def update_project_api(request, project_id):
    """POST: update project settings."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "name" in data:
        name = data["name"].strip()
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        project.name = name
        update_fields.append("name")
    if "instructions" in data:
        project.instructions = data["instructions"]
        update_fields.append("instructions")
    project.save(update_fields=update_fields)
    return JsonResponse(
        {
            "id": str(project.id),
            "name": project.name,
            "instructions": project.instructions,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def toggle_autopilot_api(request, project_id):
    """POST: toggle autopilot_enabled on a project."""
    project = get_object_or_404(Project, id=project_id)
    project.autopilot_enabled = not project.autopilot_enabled
    project.save(update_fields=["autopilot_enabled", "updated_at"])
    return JsonResponse({"autopilot_enabled": project.autopilot_enabled})


def _prompt_json(p):
    return {
        "id": str(p.id),
        "prompt_id": p.prompt_id,
        "name": p.name,
        "content": p.content,
        "built_in": p.built_in,
    }


def _slugify_prompt_id(name):
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@require_GET
def project_prompts_api(request, project_id):
    """GET: list prompts for a project, lazy-creating from registry."""
    from .prompts import prompt_registry

    project = get_object_or_404(Project, id=project_id)
    existing = {p.prompt_id for p in project.prompts.all()}
    to_create = [
        Prompt(
            project=project,
            prompt_id=pid,
            name=entry["name"],
            content=entry["content"],
            built_in=True,
        )
        for pid, entry in prompt_registry.items()
        if pid not in existing
    ]
    if to_create:
        Prompt.objects.bulk_create(to_create)
    prompts = project.prompts.order_by("built_in", "created_at")
    return JsonResponse({"prompts": [_prompt_json(p) for p in prompts]})


@csrf_exempt
@require_http_methods(["POST"])
def create_prompt_api(request, project_id):
    """POST: create a custom prompt."""
    project = get_object_or_404(Project, id=project_id)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    prompt = Prompt.objects.create(
        project=project,
        prompt_id=_slugify_prompt_id(name),
        name=name,
        content=data.get("content", ""),
        built_in=False,
    )
    return JsonResponse(_prompt_json(prompt), status=201)


@csrf_exempt
@require_http_methods(["POST"])
def update_prompt_api(request, project_id, prompt_id):
    """POST: update a prompt's content (and name for custom prompts)."""
    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    update_fields = ["updated_at"]
    if "content" in data:
        prompt.content = data["content"]
        update_fields.append("content")
    if "name" in data and not prompt.built_in:
        name = data["name"].strip()
        if name:
            prompt.name = name
            prompt.prompt_id = _slugify_prompt_id(name)
            update_fields.extend(["name", "prompt_id"])
    prompt.save(update_fields=update_fields)
    return JsonResponse(_prompt_json(prompt))


@csrf_exempt
@require_http_methods(["POST"])
def reset_prompt_api(request, project_id, prompt_id):
    """POST: reset a prompt's content to the registry default."""
    from .prompts import prompt_registry

    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    default = prompt_registry.get(prompt.prompt_id)
    if not default:
        return JsonResponse(
            {"error": "No default found for this prompt."}, status=400
        )
    prompt.content = default["content"]
    prompt.save(update_fields=["content", "updated_at"])
    return JsonResponse(_prompt_json(prompt))


@csrf_exempt
@require_http_methods(["POST"])
def delete_prompt_api(request, project_id, prompt_id):
    """POST: delete a custom prompt."""
    project = get_object_or_404(Project, id=project_id)
    prompt = get_object_or_404(Prompt, id=prompt_id, project=project)
    if prompt.built_in:
        return JsonResponse(
            {"error": "Cannot delete a built-in prompt."}, status=400
        )
    prompt.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def delete_project_api(request, project_id):
    """POST: delete a project."""
    project = get_object_or_404(Project, id=project_id)
    project.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def delete_label_api(request, project_id, label_id):
    """POST: delete a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    if project.labels.count() <= 1:
        return JsonResponse(
            {"error": "Cannot delete the only label."}, status=400
        )
    was_active = project.active_label_id == label.id
    label.delete()
    if was_active:
        new_active = project.labels.first()
        project.active_label = new_active
        project.save(update_fields=["active_label", "updated_at"])
    return JsonResponse({"ok": True})


def _task_json(t):
    from .prompts import prompt_registry

    entry = prompt_registry.get(t.prompt_id, {})
    return {
        "id": str(t.id),
        "prompt_id": t.prompt_id,
        "prompt_name": entry.get("name", t.prompt_id),
        "label": {"id": str(t.label.id), "name": t.label.name}
        if t.label
        else None,
        "status": t.status,
    }


@require_GET
def tasks_api(request, project_id):
    """GET: return current tasks without recalculating."""
    project = get_object_or_404(Project, id=project_id)
    tasks = list(project.tasks.select_related("label").order_by("created_at"))
    return JsonResponse({"tasks": [_task_json(t) for t in tasks]})


@csrf_exempt
@require_http_methods(["POST"])
def update_tasks_api(request, project_id):
    """POST: recalculate and return project tasks."""
    project = get_object_or_404(Project, id=project_id)
    tasks = project.update_tasks()
    return JsonResponse({"tasks": [_task_json(t) for t in tasks]})


@require_GET
def label_threads_api(request, project_id, label_id):
    """GET: list threads for a label with latest message preview."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    # Lazily create the autopilot thread.
    if not label.autopilot_thread_id:
        autopilot = Thread.objects.create(label=label)
        label.autopilot_thread = autopilot
        label.save(update_fields=["autopilot_thread", "updated_at"])
    autopilot_id = str(label.autopilot_thread_id)
    threads = label.threads.order_by("-updated_at")
    result = []
    for t in threads:
        latest = t.messages.order_by("created_at").first()
        preview = ""
        if latest and latest.data.get("content"):
            preview = latest.data["content"][:100]
        result.append(
            {
                "id": str(t.id),
                "model": t.model,
                "preview": preview,
                "updated_at": t.updated_at.isoformat(),
                "is_autopilot": str(t.id) == autopilot_id,
                "autopilot_locked": t.autopilot_locked,
            }
        )
    # Sort autopilot thread first.
    result.sort(key=lambda t: (not t["is_autopilot"], t["updated_at"]))
    return JsonResponse(
        {"threads": result, "autopilot_thread_id": autopilot_id}
    )


@csrf_exempt
@require_http_methods(["POST"])
def create_thread_api(request, project_id, label_id):
    """POST: create a new thread for a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}
    model = data.get("model", django_settings.DEFAULT_MODEL)
    thread = Thread.objects.create(label=label, model=model)
    return JsonResponse(
        {"id": str(thread.id), "model": thread.model}, status=201
    )


def _active_token_count(thread, messages=None):
    """Sum num_tokens from the last compact point onward, excluding hidden."""
    compact_msg = (
        thread.messages.filter(is_compact=True)
        .order_by("-created_at")
        .values_list("created_at", flat=True)
        .first()
    )
    qs = thread.messages.filter(hidden=False)
    if compact_msg:
        qs = qs.filter(created_at__gte=compact_msg)
    return qs.aggregate(total=Sum("num_tokens"))["total"] or 0


@require_GET
def thread_messages_api(request, project_id, thread_id):
    """GET: list messages for a thread."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    messages = list(thread.messages.order_by("created_at"))
    total_tokens = _active_token_count(thread, messages)
    return JsonResponse(
        {
            "messages": [
                {
                    "id": str(m.id),
                    "data": m.data,
                    "num_tokens": m.num_tokens,
                    "is_compact": m.is_compact,
                }
                for m in messages
            ],
            "usage": {
                "total_tokens": total_tokens,
                "total_cost": thread.total_cost,
            },
            "autopilot_locked": thread.autopilot_locked,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def delete_thread_api(request, project_id, thread_id):
    """POST: delete a thread."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    thread.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_http_methods(["POST"])
def recalculate_prediction_stats_api(request, project_id, label_id):
    """POST: recalculate prediction stats from existing predictions."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    label.recalculate_prediction_stats()
    return JsonResponse({"prediction_stats": label.prediction_stats})


@csrf_exempt
@require_http_methods(["POST"])
def predict_label_api(request, project_id, label_id):
    """POST: run predictions for a label using its finetuned model."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    if label.finetune_status != "completed":
        return JsonResponse(
            {"error": "No completed finetune available."}, status=400
        )
    try:
        label.predict()
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse(
        {
            "status": "ok",
            "predicted_at": label.predicted_at.isoformat()
            if label.predicted_at
            else None,
            "prediction_stats": label.prediction_stats,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def cancel_finetune_api(request, project_id, label_id):
    """POST: cancel an in-progress finetune job."""
    import os

    import requests as http_requests

    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    if label.finetune_status != "in_progress":
        return JsonResponse({"error": "No finetune in progress."}, status=400)

    endpoint_id = os.getenv("RUNPOD_FINETUNE_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    if endpoint_id and api_key and label.finetune_id:
        try:
            http_requests.post(
                f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{label.finetune_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except Exception:
            pass

    label.finetune_status = "error"
    label.save(update_fields=["finetune_status", "updated_at"])
    return JsonResponse({"status": "error"})


@csrf_exempt
@require_http_methods(["POST"])
def finetune_label_api(request, project_id, label_id):
    """POST: kick off a finetuning job for a label."""
    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)
    if label.finetune_status == "in_progress":
        return JsonResponse(
            {"error": "Finetune already in progress."}, status=409
        )
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}
    training_args = data.get("training_args") or None
    try:
        job_id = label.finetune(training_args=training_args)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"job_id": job_id, "status": label.finetune_status})


@require_GET
def finetune_status_api(request, project_id, label_id):
    """GET: check finetune status for a label."""
    import os

    import requests as http_requests

    project = get_object_or_404(Project, id=project_id)
    label = get_object_or_404(Label, id=label_id, project=project)

    if label.finetune_status != "in_progress" or not label.finetune_id:
        # Find latest annotation updated_at for staleness check.
        from .models import ClassificationAnnotation

        latest_annotation = (
            ClassificationAnnotation.objects.filter(
                label_document__label=label,
                source="agent",
                value__in=["yes", "no"],
            )
            .order_by("-updated_at")
            .values_list("updated_at", flat=True)
            .first()
        )
        return JsonResponse(
            {
                "status": label.finetune_status,
                "finetuned_at": label.finetuned_at.isoformat()
                if label.finetuned_at
                else None,
                "training_args": label.finetune_training_args,
                "latest_annotation_at": latest_annotation.isoformat()
                if latest_annotation
                else None,
            }
        )

    # Poll RunPod for job status.
    endpoint_id = os.getenv("RUNPOD_FINETUNE_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    progress = None

    if endpoint_id and api_key:
        try:
            resp = http_requests.get(
                f"https://api.runpod.ai/v2/{endpoint_id}/status/{label.finetune_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            status_data = resp.json()

            if status_data["status"] == "COMPLETED":
                from django.utils import timezone

                label.finetune_status = "completed"
                label.finetuned_at = timezone.now()
                label.save(
                    update_fields=[
                        "finetune_status",
                        "finetuned_at",
                        "updated_at",
                    ]
                )
            elif status_data["status"] in ("FAILED", "CANCELLED"):
                label.finetune_status = "error"
                label.save(update_fields=["finetune_status", "updated_at"])
            else:
                progress = status_data.get("output", {})
        except Exception:
            # RunPod unreachable or job expired — try the pipeline.
            try:
                label.pipe.predict(["test"], batch_size=1)
                from django.utils import timezone

                label.finetune_status = "completed"
                label.finetuned_at = timezone.now()
                label.save(
                    update_fields=[
                        "finetune_status",
                        "finetuned_at",
                        "updated_at",
                    ]
                )
            except Exception:
                pass

    from .models import ClassificationAnnotation

    latest_annotation = (
        ClassificationAnnotation.objects.filter(
            label_document__label=label,
            source="agent",
            value__in=["yes", "no"],
        )
        .order_by("-updated_at")
        .values_list("updated_at", flat=True)
        .first()
    )

    return JsonResponse(
        {
            "status": label.finetune_status,
            "finetuned_at": label.finetuned_at.isoformat()
            if label.finetuned_at
            else None,
            "training_args": label.finetune_training_args,
            "progress": progress,
            "latest_annotation_at": latest_annotation.isoformat()
            if latest_annotation
            else None,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def send_message_api(request, project_id, thread_id):
    """POST: send a user message and get an agent response."""
    project = get_object_or_404(Project, id=project_id)
    thread = get_object_or_404(Thread, id=thread_id, label__project=project)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    content = data.get("content", "").strip()
    if not content:
        return JsonResponse({"error": "content is required"}, status=400)
    if thread.autopilot_locked:
        return JsonResponse(
            {"error": "This thread is currently being used by autopilot."},
            status=423,
        )

    from clx.llm.agent import message_tokens

    from .agent import CLXAgent
    from .models import Message as MessageModel

    # If this is an autopilot thread, just save the message — don't run the agent.
    is_autopilot = (
        hasattr(thread.label, "autopilot_thread")
        and thread.label.autopilot_thread_id == thread.id
    )
    if is_autopilot:
        msg_data = {"role": "user", "content": content}
        MessageModel.objects.create(
            thread=thread,
            data=msg_data,
            num_tokens=message_tokens(msg_data),
        )
        # Mark any awaiting_input task for this label as pending.
        Task.objects.filter(
            label=thread.label,
            status=Task.Status.AWAITING_INPUT,
        ).update(status=Task.Status.PENDING)
        total_tokens = _active_token_count(thread)
        return JsonResponse(
            {
                "messages": [],
                "usage": {
                    "total_tokens": total_tokens,
                    "total_cost": thread.total_cost,
                },
            }
        )

    agent = CLXAgent(thread)
    msg_count_before = len(agent.messages)
    agent.run(content)
    # Return all new messages (skipping the user message already shown)
    new_messages = [
        m
        for m in agent.messages[msg_count_before + 1 :]
        if m.get("role") != "system"
    ]

    thread.refresh_from_db(fields=["total_cost"])
    total_tokens = _active_token_count(thread)
    return JsonResponse(
        {
            "messages": new_messages,
            "usage": {
                "total_tokens": total_tokens,
                "total_cost": thread.total_cost,
            },
        }
    )
