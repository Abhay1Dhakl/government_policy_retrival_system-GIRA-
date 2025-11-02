from rest_framework import serializers
from ..models import User


class PasswordSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def create(self, validated_data):
        user = User.objects.get(email=validated_data["email"])
        user.set_password(validated_data["password"])
        user.is_active = True
        user.save()
        return user
