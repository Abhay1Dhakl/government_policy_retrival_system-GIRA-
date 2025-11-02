from typing import Any, Dict

from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler


def custom_exception_handler(exc: Exception, context: Dict[str, Any]) -> Response:
    response = exception_handler(exc, context)

    if isinstance(exc, ValidationError):
        custom_response_data = {
            "status": status.HTTP_400_BAD_REQUEST,
            "message": "Validation error",
            "errors": exc.detail,
        }
        return Response(custom_response_data, status=status.HTTP_400_BAD_REQUEST)

    if response is not None:
        response.data = {
            "status": response.status_code,
            "message": response.data.get("detail", "Unknown error"),
        }

    return response


def api_response(
    data: Any = None,
    message: str = "Success",
    status_code: int = 200,
) -> Response:
    response_data = {
        "status": status_code,
        "message": message,
        "data": data,
    }
    return Response(response_data, status=status_code)
