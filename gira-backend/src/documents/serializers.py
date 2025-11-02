import json
import os
import requests
from rest_framework import serializers
from .models import Document, DocumentFile
from .services.save_file import fetch_and_save_file
from src.constants.database_constants import FileSource
from .utils import upload_document
import logging

logger = logging.getLogger(__name__)

class DocumentFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentFile
        fields = "__all__"
        read_only_fields = ["id", "document"]


class DocumentSerializer(serializers.ModelSerializer):
    files = DocumentFileSerializer(many=True, read_only=True)
    file = serializers.FileField(write_only=True, required=False)

    class Meta:
        model = Document
        fields = "__all__"

    def create(self, validated_data):
        file_obj = validated_data.pop("file", None)
        file_source = validated_data.get("file_source")
        file_source_link = validated_data.get("file_source_link")

        if file_source and file_source != FileSource.UPLOAD:
            file_obj = fetch_and_save_file(file_source_link, file_source)

        document = Document.objects.create(**validated_data)

        filename = file_obj.name

        get_existence = Document.objects.filter(minio_object_name=filename).exists()

        if get_existence:
            base, extension = os.path.splitext(filename)
            count = 1
            new_filename = f"{base}_{count}{extension}"
            while Document.objects.filter(minio_object_name=new_filename).exists():
                count += 1
                new_filename = f"{base}_{count}{extension}"
            object_name = new_filename
        else:
            object_name = filename

        document.minio_object_name = object_name
        print(f"Assigned minio_object_name: {document.minio_object_name}")
        document.save()

        # Get ingest URL from environment or use default
        ingest_base_url = os.getenv("INGEST_BASE_URL", "http://mira-agent:8081")
        url = f"{ingest_base_url}/admin/documents/ingest/"
        payload = {
            "instance_id": validated_data.get("instance_id"),
            "document_type": validated_data.get("document_type"),
            "source_type": validated_data.get("source_type").lower(),
            "document_metadata": {
                "title": validated_data.get("title"),
                "language": validated_data.get("language"),
                "region": validated_data.get("region"),
                "author": validated_data.get("author"),
                "tags": (
                    validated_data.get("tags").split(",")
                    if validated_data.get("tags")
                    else []
                ),
            },
            "api_connection_info": {
                "auth_type": validated_data.get("auth_type"),
                "client_id": validated_data.get("client_id"),
                "client_secret": validated_data.get("client_secret"),
                "token_url": validated_data.get("token_url"),
                "data_url": validated_data.get("data_url"),
            },
            "manual_text": validated_data.get("manual_text"),
            "document_id": str(document.id),
            "file_name": object_name,
        }

        form_data = {"data": json.dumps(payload)}
        files = {"file_upload": file_obj} if file_obj else None

        try:
            response = requests.post(url, data=form_data, files=files, timeout=30)

            if response.status_code != 200:
                error_msg = f"Failed to ingest document. Status: {response.status_code}, Response: {response.text}"
                raise serializers.ValidationError(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to ingest service: {e}"
            raise serializers.ValidationError(error_msg)

        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout calling ingest service: {e}"
            raise serializers.ValidationError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error calling ingest service: {e}"
            raise serializers.ValidationError(error_msg)

        # Upload to MinIO with sequential filename
        if file_obj:
            minio_object_name = document.minio_object_name or f"{document.id}.pdf"
            upload_document(file_obj, object_name=minio_object_name)

        # Clean up local file
        if file_obj and os.path.exists(file_obj.name):
            os.remove(file_obj.name)

        return document
