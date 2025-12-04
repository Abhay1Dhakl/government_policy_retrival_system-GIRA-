from typing import Any

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet


from src.gira.utils.response import api_response
from ..serializers.password import PasswordSerializer
from ..models import User


class PasswordViewSet(GenericViewSet):
    serializer_class = PasswordSerializer
    queryset = User.objects.all()

    def get_queryset(self):
        return super().get_queryset()

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = get_object_or_404(User, email=request.data.get("email"))

        if user.password and user.password != "":
            return api_response(
                data=None,
                message="Password has already been set",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        user = serializer.save()

        return api_response(
            data={"message": "Password updated successfully", "user_id": user.id},
            status_code=status.HTTP_200_OK,
        )
