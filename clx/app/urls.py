from django.urls import path

from . import views

urlpatterns = [
    path("", views.index_view),
    path("<slug:project_slug>/", views.search_view, name="search"),
    path(
        "api/search/<slug:project_slug>/",
        views.search_endpoint,
        name="search-endpoint",
    ),
]
