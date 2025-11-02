from typing import Any

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from ..models import User
from ..serializers.user import UserSerializer, UserCreateSerializer
from src.mira.utils.response import api_response


class UserViewSet(GenericViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()
    permission_classes = [IsAuthenticated, IsAdminUser]

    def get_queryset(self):
        return super().get_queryset().filter(is_active=True).order_by("-created_at")

    def get_permissions(self):
        if self.action == "generate_token":
            self.permission_classes = [AllowAny]
        elif self.action == "update_user" or self.action == "user_details":
            self.permission_classes = [IsAuthenticated]
        return [permission() for permission in self.permission_classes]

    def get_serializer_class(self):
        if self.action == "create":
            return UserCreateSerializer
        else:
            return self.serializer_class

    def list(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        users = self.get_queryset()

        if not users:
            return api_response(
                data=[],
                message="No users found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        serializer = UserSerializer(users, many=True)
        return api_response(
            data=serializer.data,
            message="User retrieved successfully",
            status_code=status.HTTP_200_OK,
        )

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return api_response(
            data=serializer.data,
            message="User created successfully",
            status_code=status.HTTP_201_CREATED,
        )

    @action(detail=False, methods=["patch"], url_path="update")
    def update_user(self, request: Request) -> Response:
        user = request.user
        serializer = self.get_serializer(user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        updated_user = serializer.save()

        return api_response(
            data=UserSerializer(updated_user).data,
            message="User updated successfully",
            status_code=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["get"], url_path="details")
    def user_details(self, request: Request) -> Response:
        user = request.user
        serializer = self.get_serializer(user)
        return api_response(
            data=serializer.data,
            message="User details retrieved successfully",
            status_code=status.HTTP_200_OK,
        )
