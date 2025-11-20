import logging

from django.conf import settings
from django.http import JsonResponse

logger = logging.getLogger(__name__)


class ApiExceptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exc):
        """Return a JSON response for API exceptions."""
        if request.path.startswith("/api/"):
            logger.exception("Unhandled exception in API view", exc_info=exc)
            error = str(exc) if settings.DEBUG else "Internal server error"
            return JsonResponse({"error": error}, status=500)
