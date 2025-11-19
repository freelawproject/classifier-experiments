import json

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import Label, LabelTag, Project


@csrf_exempt
@require_POST
def search_endpoint(request, project_id):
    try:
        payload = {} if request.body is None else json.loads(request.body)
        # payload['params']['tags'] = {"any": ["scales:motion"]}
        project = Project.objects.get(id=project_id)
        model = project.get_search_model()
        return JsonResponse(model.objects.search(**payload))
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def search_view(request, project_id):
    project = Project.objects.get(id=project_id)

    labels = Label.objects.filter(project=project)
    labels = labels.values("id", "name")
    labels = {label["id"]: label for label in labels}

    tags = LabelTag.objects.filter(label__project=project)
    tags = tags.values("id", "label_id", "slug")
    tags = {tag["id"]: tag for tag in tags}
    return render(
        request,
        "search.html",
        {
            "project": project,
            "projects": Project.objects.all().order_by("name"),
            "labels": json.dumps(labels, default=str),
            "tags": json.dumps(tags, default=str),
        },
    )


def index_view(request):
    return redirect("search", project_id="docket-entry")
