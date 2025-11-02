from django.db import models

from src.constants.database_constants import DocumentType, FileSource, SourceType


class Document(models.Model):
    class Meta:
        db_table = "documents"

    instance_id = models.CharField(max_length=255, unique=True)
    document_type = models.CharField(max_length=50, choices=DocumentType.choices)
    source_type = models.CharField(max_length=50, choices=SourceType.choices)
    title = models.CharField(max_length=255)
    language = models.CharField(max_length=10)
    region = models.CharField(max_length=50, null=True, blank=True)
    author = models.CharField(max_length=255, null=True, blank=True)
    tags = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_global = models.BooleanField(default=False)
    file_source = models.CharField(
        max_length=50, choices=FileSource.choices, null=True, blank=True
    )
    file_source_link = models.URLField(null=True, blank=True)
    minio_object_name = models.CharField(max_length=255, null=True, blank=True)
    is_deleted = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.title} ({self.document_type})"


class DocumentFile(models.Model):
    class Meta:
        db_table = "document_files"

    document = models.ForeignKey(
        Document, related_name="files", on_delete=models.CASCADE
    )
    file = models.FileField(upload_to="documents/")
