import os

from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"
# MINIO_PUBLIC_URL = "localhost:9000"
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "93.127.143.185:9000")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


def upload_document(file_data, object_name=None, bucket_name="medical-documents"):
    if not object_name:
        filename = os.path.basename(file_data.name)
        _, ext = os.path.splitext(filename)
        object_name = filename if ext else f"{filename}.bin"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    file_data.seek(0, os.SEEK_END)
    file_size = file_data.tell()
    file_data.seek(0)
    minio_client.put_object(
        bucket_name,
        object_name,
        file_data,
        file_size,
    )
    return f"{bucket_name}/{object_name}"


def get_minio_public_url(bucket_name, object_name):
    """Generate public URL for MinIO object"""
    protocol = "https" if MINIO_SECURE else "http"
    return f"{protocol}://{MINIO_PUBLIC_URL}/{bucket_name}/{object_name}"
