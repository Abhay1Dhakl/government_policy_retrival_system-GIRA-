from rest_framework import serializers
from ..models import User

from src.mira_emails.tasks import send_invite_mail


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"
        read_only_fields = (
            "id",
            "created_at",
            "updated_at",
            "last_login",
            "is_superuser",
            "role",
            "is_active",
            "is_staff",
            "groups",
            "user_permissions",
            "email",
            "password",
        )

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation.pop("password", None)
        return representation

    def update(self, instance, validated_data):
        validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance


class UserCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["email"]

    def create(self, validated_data):
        print(
            f"**************** Creating user with data: {validated_data} ****************"
        )
        user = User.objects.create_user(**validated_data)
        send_invite_mail.delay(user.email, user.first_name)
        return user
