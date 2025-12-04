from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from ..models import User

from src.gira_emails.tasks import send_invite_mail


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


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, style={"input_type": "password"})
    password_confirm = serializers.CharField(write_only=True, required=True, style={"input_type": "password"})

    class Meta:
        model = User
        fields = ["email", "password", "password_confirm", "first_name", "last_name"]

    def validate(self, attrs):
        # Check if passwords match
        if attrs["password"] != attrs["password_confirm"]:
            raise serializers.ValidationError({"password_confirm": "Passwords do not match."})
        
        # Validate password strength using Django's built-in validators
        try:
            validate_password(attrs["password"])
        except ValidationError as e:
            raise serializers.ValidationError({"password": list(e.messages)})
        
        return attrs

    def create(self, validated_data):
        # Remove password_confirm as it's not needed for user creation
        validated_data.pop("password_confirm")
        
        # Create user with is_active=True (no invitation needed)
        user = User.objects.create_user(
            email=validated_data["email"],
            password=validated_data["password"],
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
        )
        user.is_active = True
        user.save()
        
        return user
