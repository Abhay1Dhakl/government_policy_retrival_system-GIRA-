from django.http import FileResponse
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.viewsets import GenericViewSet
from src.gira.utils.response import api_response

from .models import Document
from .serializers import DocumentSerializer


class DocumentViewSet(GenericViewSet):
    serializer_class = DocumentSerializer
    queryset = Document.objects.all()
    parser_classes = (MultiPartParser, FormParser)

    def get_serializer(self, *args, **kwargs):
        return DocumentSerializer(*args, **kwargs)

    def get_permissions(self):
        if self.action == "download":
            permission_classes = [AllowAny]
        else:
            permission_classes = [IsAdminUser]
        return [permission() for permission in permission_classes]

    def list(self, request, *args, **kwargs):
        documents = self.get_queryset()
        serializer = self.get_serializer(documents, many=True)
        return api_response(
            data=serializer.data,
            message="Documents retrieved successfully",
            status_code=status.HTTP_200_OK,
        )

    @swagger_auto_schema(
        operation_description="Create a new document with optional file uploads",
        request_body=DocumentSerializer,
        responses={201: DocumentSerializer, 400: "Bad Request"},
    )
    def create(self, request, *args, **kwargs):
        file = request.FILES.get("file")

        data = request.data.copy()
        if file:
            data["file"] = file

        try:
            serializer = DocumentSerializer(data=data, context={"request": request})
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return api_response(
                data=serializer.data,
                message="Document created successfully",
                status_code=status.HTTP_201_CREATED,
            )

        except Exception as e:
            return api_response(
                data=str(e),
                message="Invalid data",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        document = self.get_object()
        file_instance = document.files.first()
        if not file_instance:
            return api_response(
                data=None,
                message="No file associated with this document",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        file_handle = file_instance.file.open()
        response = FileResponse(
            file_handle, as_attachment=True, filename=file_instance.file.name
        )
        return response
