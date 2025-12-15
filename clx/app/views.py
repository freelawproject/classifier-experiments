import inspect
import json

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .custom_heuristics import custom_heuristics
from .models import Label, LabelDecision, LabelHeuristic, LabelTag, Project


# Endpoints
## Search Endpoints
@csrf_exempt
@require_POST
def search_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    project = Project.objects.get(id=project_id)
    model = project.get_search_model()
    return JsonResponse(model.objects.search(**payload))


# Project Endpoints
@require_GET
def project_endpoint(request, project_id):
    project = Project.objects.get(id=project_id)
    print(project)
    return JsonResponse(
        {
            "project": {
                "id": project.id,
                "name": project.name,
                "instructions": project.instructions or "",
            }
        }
    )


@csrf_exempt
@require_POST
def project_update_instructions_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    instructions = payload.get("instructions", "")
    project = Project.objects.get(id=project_id)
    project.instructions = instructions
    project.save()
    return JsonResponse(
        {"ok": True, "instructions": project.instructions or ""}
    )


# Labels Endpoints
@require_GET
def labels_endpoint(request, project_id):
    project = Project.objects.get(id=project_id)
    labels_qs = Label.objects.filter(project=project).values(
        "id",
        "name",
        "num_excluded",
        "num_neutral",
        "num_likely",
        "instructions",
        "inference_model",
        "teacher_model",
        "predictor_data",
        "predictor_updated_at",
        "trainset_num_excluded",
        "trainset_num_neutral",
        "trainset_num_likely",
        "trainset_num_positive_preds",
        "trainset_num_negative_preds",
        "trainset_predictions_updated_at",
        "trainset_updated_at",
    )
    labels = {row["id"]: row for row in labels_qs}
    return JsonResponse({"labels": labels})


@csrf_exempt
@require_POST
def labels_update_instructions_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    instructions = payload.get("instructions", "")
    label = Label.objects.get(id=label_id, project_id=project_id)
    label.instructions = instructions
    label.save()
    return JsonResponse({"ok": True})


# Tags Endpoints
@require_GET
def tags_endpoint(request, project_id):
    project = Project.objects.get(id=project_id)
    tags_qs = LabelTag.objects.filter(label__project=project).values(
        "id", "label_id", "slug"
    )
    tags = {row["id"]: row for row in tags_qs}
    return JsonResponse({"tags": tags})


@csrf_exempt
@require_POST
def annotate_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    example_id = payload.get("example_id")
    label_id = payload.get("label_id")
    value = payload.get("value", None)
    project = Project.objects.get(id=project_id)
    model = project.get_search_model()
    example = model.objects.get(id=example_id)
    label = Label.objects.get(id=label_id, project_id=project_id)
    example.set_annotation(label, value)
    return JsonResponse({"ok": True})


# Decision Endpoints
@csrf_exempt
@require_POST
def decisions_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    assert label_id, "label_id is required"
    decisions = LabelDecision.objects.filter(label_id=label_id).values(
        "id",
        "label_id",
        "value",
        "reason",
        "text_hash",
        "text",
        "updated_at",
    )
    decisions = {d["id"]: d for d in decisions}
    return JsonResponse({"decisions": decisions})


@csrf_exempt
@require_POST
def decision_update_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    text_hash = payload.get("text_hash")
    value = payload.get("value")
    reason = payload.get("reason")
    print(label_id, text_hash, value, reason)
    assert label_id and isinstance(value, bool) and reason, (
        "label_id, value and reason are required"
    )
    label = Label.objects.get(id=label_id)
    decision, _ = LabelDecision.objects.get_or_create(
        label_id=label.id,
        text_hash=text_hash,
        defaults={"value": value, "reason": reason},
    )
    if decision.text is None:
        decision.text = (
            label.project.get_search_model()
            .objects.get(text_hash=text_hash)
            .text
        )
    decision.value = value
    decision.reason = reason
    decision.save()
    data = {
        "id": decision.id,
        "label_id": decision.label_id,
        "text_hash": decision.text_hash,
        "value": decision.value,
        "reason": decision.reason,
        "text": decision.text,
    }
    return JsonResponse({"decision": data})


@csrf_exempt
@require_POST
def decision_delete_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    decision_id = payload.get("decision_id")
    if not decision_id:
        return JsonResponse({"error": "decision_id is required"}, status=400)
    decision = LabelDecision.objects.get(id=decision_id)
    decision.delete()
    return JsonResponse({"ok": True})


## Heuristic Endpoints
@csrf_exempt
@require_POST
def heuristics_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    assert label_id, "label_id is required"
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


# Predictor Endpoints
@csrf_exempt
@require_POST
def predictor_update_trainset_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    assert label_id, "label_id is required"
    label = Label.objects.get(id=label_id)
    label.trainset_num_excluded = int(
        payload.get("trainset_num_excluded", 1000)
    )
    label.trainset_num_neutral = int(payload.get("trainset_num_neutral", 1000))
    label.trainset_num_likely = int(payload.get("trainset_num_likely", 1000))
    label.save()
    label.update_trainset()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def predictor_update_trainset_preds_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    assert label_id, "label_id is required"
    label = Label.objects.get(id=label_id)
    label.update_trainset_preds()
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def predictor_fit_endpoint(request, project_id):
    payload = {} if request.body is None else json.loads(request.body)
    label_id = payload.get("label_id")
    assert label_id, "label_id is required"
    label = Label.objects.get(id=label_id)
    label.inference_model = payload.get("inference_model")
    label.teacher_model = payload.get("teacher_model")
    label.save()
    label.fit_predictor()
    return JsonResponse({"ok": True})


# Views
def search_view(request, project_id):
    project = Project.objects.get(id=project_id)
    return render(
        request,
        "search/index.html",
        {
            "project": project,
            "projects": Project.objects.all().order_by("name"),
            "label_class": Label,
        },
    )


def index_view(request):
    return redirect("search", project_id="docket-entry")
