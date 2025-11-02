import os
from cryptography.fernet import Fernet

from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from ..models import User
from ui.models import LLM


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        active_model = LLM.objects.filter(is_active=True).first()
        if active_model:
            token["llm_model"] = {
                "name": active_model.name,
                "model": active_model.model,
                "api_key": active_model.api_key,
            }
        token["country"] = user.country

        return token


class UserTokenSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate(self, attrs):
        user = User.objects.filter(email=attrs["email"]).first()
        if not user.password:
            raise serializers.ValidationError("User has not set a password")
        if not user or not user.check_password(attrs["password"]):
            raise serializers.ValidationError("Invalid email or password")
        return attrs


class RefreshTokenSerializer(serializers.Serializer):
    refresh_token = serializers.CharField(required=True)

    def validate(self, attrs):
        if not attrs.get("refresh_token"):
            raise serializers.ValidationError("Refresh token is required")
        return attrs


class OAuthCallbackSerializer(serializers.Serializer):
    token = serializers.CharField(required=True)
