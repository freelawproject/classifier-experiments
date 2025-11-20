import inspect
import json

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .custom_heuristics import custom_heuristics
from .models import Label, LabelHeuristic, LabelTag, Project


# Endpoints
## Search Endpoints
@csrf_exempt
@require_POST
def search_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    project = Project.objects.get(id=project_id)
    model = project.get_search_model()
    return JsonResponse(model.objects.search(**payload))


@require_GET
def labels_endpoint(request, project_id):
    project = Project.objects.get(id=project_id)
    labels_qs = Label.objects.filter(project=project).values(
        "id", "name", "num_excluded", "num_neutral", "num_likely"
    )
    labels = {row["id"]: row for row in labels_qs}
    return JsonResponse({"labels": labels})


@require_GET
def tags_endpoint(request, project_id):
    project = Project.objects.get(id=project_id)
    tags_qs = LabelTag.objects.filter(label__project=project).values(
        "id", "label_id", "slug"
    )
    tags = {row["id"]: row for row in tags_qs}
    return JsonResponse({"tags": tags})


## Heuristic Endpoints
@require_GET
def heuristics_endpoint(request, project_id, label_id):
    heuristics = list(
        LabelHeuristic.objects.filter(label_id=label_id).values(
            "id",
            "querystring",
            "custom",
            "applied_at",
            "created_at",
            "is_minimal",
            "is_likely",
            "num_examples",
        )
    )
    for heuristic in heuristics:
        if heuristic["custom"] is not None:
            try:
                heuristic["source"] = inspect.getsource(
                    custom_heuristics[heuristic["custom"]]["apply_fn"]
                )
            except Exception:
                heuristic["source"] = (
                    "This heuristic has been deleted. Please sync custom heuristics."
                )
    heuristics = {h["id"]: h for h in heuristics}
    return JsonResponse({"heuristics": heuristics})


@csrf_exempt
@require_POST
def heuristic_add_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    querystring = (payload.get("querystring") or "").strip()
    if not label_id or not querystring:
        return JsonResponse(
            {"error": "label_id and querystring are required"}, status=400
        )
    heuristic = LabelHeuristic.objects.create(
        label_id=label_id,
        querystring=querystring,
    )
    return JsonResponse({"ok": True, "id": heuristic.id})


@csrf_exempt
@require_POST
def heuristic_apply_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    heuristic_id = payload.get("heuristic_id")
    if not heuristic_id:
        raise ValueError("heuristic_id is required")
    heuristic = LabelHeuristic.objects.get(id=heuristic_id)
    heuristic.apply()
    return JsonResponse({"ok": True, "applied_at": heuristic.applied_at})


@csrf_exempt
@require_POST
def heuristic_set_flag_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    heuristic_id = payload.get("heuristic_id")
    flag = payload.get("flag")
    if not heuristic_id or flag not in {"is_minimal", "is_likely"}:
        return JsonResponse(
            {"error": "heuristic_id and valid flag are required"}, status=400
        )
    heuristic = LabelHeuristic.objects.get(id=heuristic_id)
    setattr(heuristic, flag, not getattr(heuristic, flag))
    heuristic.save()
    return JsonResponse({"ok": True, flag: getattr(heuristic, flag)})


@csrf_exempt
@require_POST
def heuristic_delete_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    heuristic_id = payload.get("heuristic_id")
    if not heuristic_id:
        raise ValueError("heuristic_id is required")
    heuristic = LabelHeuristic.objects.get(id=heuristic_id)
    heuristic.delete()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def heuristics_sync_custom_endpoint(request, project_id):
    LabelHeuristic.sync_custom_heuristics()
    return JsonResponse({"ok": True})


# Views
def search_view(request, project_id):
    project = Project.objects.get(id=project_id)
    return render(
        request,
        "search.html",
        {
            "project": project,
            "projects": Project.objects.all().order_by("name"),
        },
    )


def index_view(request):
    return redirect("search", project_id="docket-entry")
