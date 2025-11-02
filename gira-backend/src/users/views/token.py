import os
from typing import Any

from google.oauth2 import id_token
from google.auth.transport import requests

from django.http import HttpResponse
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from rest_framework.decorators import action
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

from src.mira.utils.response import api_response
from ..serializers.token import (
    UserTokenSerializer,
    RefreshTokenSerializer,
    CustomTokenObtainPairSerializer,
    OAuthCallbackSerializer,
)
from ..models import User


class TokenViewSet(GenericViewSet):
    serializer_class = UserTokenSerializer
    queryset = User.objects.all()

    def get_serializer(self, *args, **kwargs) -> Any:
        if self.action == "create":
            return UserTokenSerializer(*args, **kwargs)
        elif self.action == "refresh_token":
            return RefreshTokenSerializer(*args, **kwargs)
        elif self.action == "oauth_callback":
            return OAuthCallbackSerializer(*args, **kwargs)

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return api_response(
                data=serializer.errors,
                message="Invalid data",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        user = User.objects.get(email=serializer.validated_data["email"])
        refresh = CustomTokenObtainPairSerializer.get_token(user)

        data = {
            "refresh_token": str(refresh),
            "access_token": str(refresh.access_token),
            "has_details": user.has_details,
        }
        return api_response(
            data=data,
            message="Token generated successfully",
            status_code=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["post"], url_path="refresh")
    def refresh_token(self, request: Request) -> Response:
        serializer = RefreshTokenSerializer(data=request.data)
        if not serializer.is_valid():
            return api_response(
                data=serializer.errors,
                message="Invalid data",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        refresh_token_str = serializer.validated_data["refresh_token"]

        try:
            refresh = RefreshToken(refresh_token_str)
            user_id = refresh["user_id"]
            user = User.objects.get(id=user_id)
            new_access_token = str(refresh.access_token)

            return api_response(
                data={
                    "refresh_token": refresh_token_str,
                    "access_token": new_access_token,
                    "has_details": user.has_details,
                },
                message="Token refreshed successfully",
                status_code=status.HTTP_200_OK,
            )

        except TokenError as e:
            return api_response(
                data={"detail": str(e)},
                message="Invalid or expired refresh token",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        except Exception as e:
            return api_response(
                data={"detail": str(e)},
                message="An error occurred",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"], url_path="oauth/callback")
    def oauth_callback(self, request: Request) -> Response:
        try:
            token = request.data.get("token")
            idinfo = id_token.verify_oauth2_token(
                token, requests.Request(), os.getenv("OAUTH_CLIENT_ID")
            )
            print(f"idinfo is {idinfo}")

            email = idinfo["email"]

            try:

                user = User.objects.get(email=email)
                refresh = CustomTokenObtainPairSerializer.get_token(user)
                user.is_active = True
                user.save()

                data = {
                    "refresh_token": str(refresh),
                    "access_token": str(refresh.access_token),
                    "has_details": user.has_details,
                }
                return api_response(
                    data=data,
                    message="Token generated successfully",
                    status_code=status.HTTP_200_OK,
                )

            except User.DoesNotExist:
                return HttpResponse(
                    "User not found or you're not invited to the organization",
                    status=status.HTTP_404_NOT_FOUND,
                )

        except Exception as e:
            return api_response(
                data={"detail": str(e)},
                message="An error occurred during OAuth callback",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
