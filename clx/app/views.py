import json

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import Label, LabelFeature, Project


@csrf_exempt
@require_POST
def search_endpoint(request, project_slug):
    try:
        page_size = 100
        payload = {} if request.body is None else json.loads(request.body)
        page = payload.get("page", 1)
        query = payload.get("query", {})
        count = payload.get("count", False)
        project = Project.objects.get(slug=project_slug)
        SearchModel = project.get_search_model_class()
        q = SearchModel.objects.search(**query)
        if count:
            r = {"count": q.count()}
        else:
            r = {"data": list(q.values().page(page, size=page_size))}
        return JsonResponse(r)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def search_view(request, project_slug):
    project = Project.objects.get(slug=project_slug)
    labels = list(Label.objects.filter(project=project).values())
    labels = {label["id"]: label for label in labels}
    features = list(
        LabelFeature.objects.filter(label__project=project).values()
    )
    features = {feature["id"]: feature for feature in features}
    return render(
        request,
        "search.html",
        {
            "project": project,
            "labels": json.dumps(labels, default=str),
            "features": json.dumps(features, default=str),
        },
    )


def index_view(request):
    return redirect("search", project_slug="docketentry")
