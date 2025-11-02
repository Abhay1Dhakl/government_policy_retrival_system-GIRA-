import json
import os
import mimetypes
import pycountry

from cryptography.fernet import Fernet

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.crypto import get_random_string
from django.http import HttpResponse, JsonResponse
from rest_framework import status

# from rest_framework import serializers

from users.models import User
from src.constants.database_constants import FileSource, DocumentType
from src.constants.llm_constants import llm_constants
from documents.models import Document
from users.serializers.user import UserCreateSerializer
from src.documents.utils import upload_document
from documents.services.save_file import fetch_and_save_file
from documents.utils import minio_client

from .models import LLM


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, username=email, password=password)

        if user is not None:
            if user.is_superuser or user.is_staff:
                login(request, user)
                return redirect("home")
            else:
                messages.error(
                    request, "You do not have permission to access the admin panel."
                )
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required(login_url="/admin-ui/login/")
def home(request):
    return render(request, "home.html")


@login_required(login_url="/admin-ui/login/")
def users_page(request):
    if request.method == "POST":
        email = request.POST.get("email")
        serializer = UserCreateSerializer(data={"email": email})
        if serializer.is_valid():
            serializer.save()
        return redirect("users_page")

    users = User.objects.order_by("-created_at")
    return render(request, "users_page.html", {"users": users})


@login_required(login_url="/admin-ui/login/")
def user_delete(request, user_id):
    if request.method == "DELETE" or request.method == "POST":
        try:
            user = User.objects.get(id=user_id)
            user.delete()
            return redirect("users_page")
        except User.DoesNotExist:
            return HttpResponse("User not found", status=404)
    return HttpResponse("Invalid request", status=400)


@login_required(login_url="/admin-ui/login/")
def user_edit(request, user_id):

    user = get_object_or_404(User, id=user_id)
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))

            excluded_fields = {
                "id",
                "email",
                "password",
                "last_login",
                "created_at",
                "updated_at",
            }

            for attr, value in data.items():
                if attr not in excluded_fields and hasattr(user, attr):
                    setattr(user, attr, value)

            user.save()
            return JsonResponse(
                {"success": True, "message": "User updated successfully."}
            )
        except Exception as e:
            return JsonResponse(
                {"success": False, "error": str(e)}, status=status.HTTP_400_BAD_REQUEST
            )

    return HttpResponse("Invalid request", status=status.HTTP_400_BAD_REQUEST)


@login_required(login_url="/admin-ui/login/")
def documents_page(request):
    if request.method == "POST":
        instance_id = get_random_string(12)

        file_data = None
        file_path = None

        if request.FILES.get("file"):
            file_data = request.FILES["file"]

        file_source = request.POST.get("file_source")
        file_source_link = request.POST.get("file_source_link")

        if file_source and file_source != FileSource.UPLOAD:
            file_path = fetch_and_save_file(file_source_link, file_source)
            file_data = open(file_path, "rb")

        # url = f"{os.getenv('INGEST_BASE_URL')}/admin/documents/ingest/"
        # payload = {
        #     "instance_id": instance_id,
        #     "document_type": request.POST.get("document_type"),
        #     "source_type": request.POST.get("source_type", "").lower(),
        #     "document_metadata": {
        #         "title": request.POST.get("title"),
        #         "language": request.POST.get("language"),
        #         "region": request.POST.get("region"),
        #         "author": request.POST.get("author"),
        #         "tags": (
        #             request.POST.get("tags").split(",")
        #             if request.POST.get("tags")
        #             else []
        #         ),
        #     },
        #     "api_connection_info": {
        #         "auth_type": request.POST.get("auth_type"),
        #         "client_id": request.POST.get("client_id"),
        #         "client_secret": request.POST.get("client_secret"),
        #         "token_url": request.POST.get("token_url"),
        #         "data_url": request.POST.get("data_url"),
        #     },
        #     "manual_text": request.POST.get("manual_text"),
        # }

        # form_data = {"data": json.dumps(payload)}
        # files = {"file_upload": file_obj} if file_obj else None

        # response = requests.post(url, data=form_data, files=files)

        # if response.status_code != 200:
        #     raise serializers.ValidationError("Failed to ingest document")

        document = Document.objects.create(
            instance_id=instance_id,
            document_type=request.POST["document_type"],
            source_type=request.POST["source_type"],
            title=request.POST["title"],
            language=request.POST["language"],
            file_source=request.POST["file_source"],
            file_source_link=request.POST.get("file_source_link"),
            region=request.POST.get("region"),
            author=request.POST.get("author"),
            tags=request.POST.get("tags"),
        )

        allow_rename = request.POST.get("allow_rename") == "true"

        filename = file_data.name

        # Use original filename instead of document ID
        object_name = filename

        # Handle filename collisions
        from documents.models import Document as DocModel

        if DocModel.objects.filter(minio_object_name=object_name).exists():
            if not allow_rename:
                messages.error(
                    request,
                    f"A document named '{object_name}' already exists. Upload cancelled.",
                )
                return redirect("documents_page")

            # Only rename if user allowed
            base, ext = os.path.splitext(object_name)
            counter = 1
            while DocModel.objects.filter(
                minio_object_name=f"{base}_{counter}{ext}"
            ).exists():
                counter += 1
            object_name = f"{base}_{counter}{ext}"

        upload_document(file_data, object_name=object_name)
        file_data.close() if file_data and not file_data.closed else None
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        document.minio_object_name = object_name
        document.save(update_fields=["minio_object_name"])

        return redirect("documents_page")

    documents = Document.objects.filter(is_deleted=False).order_by("-created_at")
    countries = [(country.alpha_2, country.name) for country in pycountry.countries]
    languages = [
        (lang.alpha_2, lang.name)
        for lang in pycountry.languages
        if hasattr(lang, "alpha_2")
    ]

    local_choices = [
        (DocumentType.PI_LEAFLET.value, DocumentType.PI_LEAFLET.label),
        (DocumentType.LRD.value, DocumentType.LRD.label),
        (DocumentType.GRD_ESCALATION.value, DocumentType.GRD_ESCALATION.label),
        (DocumentType.ONLINE_DB.value, DocumentType.ONLINE_DB.label),
    ]

    global_choices = [
        (DocumentType.PI_LEAFLET.value, DocumentType.PI_LEAFLET.label),
        (DocumentType.GRD_ESCALATION.value, DocumentType.GRD_ESCALATION.label),
        (DocumentType.ONLINE_DB.value, DocumentType.ONLINE_DB.label),
    ]

    return render(
        request,
        "documents_page.html",
        {
            "documents": documents,
            "countries": countries,
            "languages": languages,
            "local_choices": json.dumps(local_choices),
            "global_choices": json.dumps(global_choices),
        },
    )


def document_download(request, doc_id):
    try:
        action = request.GET.get("action", "download")
        if action not in ["download", "view"]:
            return HttpResponse("Invalid action", status=400)
        doc = Document.objects.get(id=doc_id)
        object_name = doc.minio_object_name

        local_dir = "/tmp/downloaded_documents"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(object_name))

        minio_client.fget_object("medical-documents", object_name, local_path)

        with open(local_path, "rb") as f:
            file_data = f.read()

        if action == "view":
            content_type, _ = mimetypes.guess_type(object_name)
            if content_type is None:
                content_type = "application/octet-stream"

            response = HttpResponse(file_data, content_type=content_type)
            response["Content-Disposition"] = (
                f'inline; filename="{os.path.basename(object_name)}"'
            )
            return response

        response = HttpResponse(file_data, content_type="application/octet-stream")
        response["Content-Disposition"] = (
            f'attachment; filename="{os.path.basename(object_name)}"'
        )
        return response

    except Document.DoesNotExist:
        return HttpResponse("Document not found", status=404)
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}", status=500)


@login_required(login_url="/admin-ui/login/")
def document_delete(request, doc_id):
    if request.method == "POST":
        try:
            doc = Document.objects.get(id=doc_id)
            doc.is_deleted = True
            doc.save(update_fields=["is_deleted"])

            # minio_object_name = doc.minio_object_name
            # if minio_object_name:
            #     url = f"{os.getenv('INGEST_BASE_URL')}/delete_all_pinecone_records/{minio_object_name}/"
            #     response = requests.delete(url)
            #     if response.status_code != 200:
            #         print("Failed to delete document from Minio")

            return redirect("documents_page")
        except Document.DoesNotExist:
            return HttpResponse("Document not found", status=404)
    return HttpResponse("Invalid request", status=400)


@login_required(login_url="/admin-ui/login/")
def check_duplicate_document(request):
    filename = request.GET.get("filename")
    if not filename:
        return JsonResponse({"exists": False})
    exists = Document.objects.filter(minio_object_name=filename).exists()

    counter = 1
    expected_filename = filename
    base, ext = os.path.splitext(filename)
    while Document.objects.filter(minio_object_name=f"{base}_{counter}{ext}").exists():
        counter += 1
        exists = True
        expected_filename = f"{base}_{counter}{ext}"

    return JsonResponse({"exists": exists, "expected_filename": expected_filename})


@login_required(login_url="/admin-ui/login/")
def llm_page(request):
    activate_id = request.GET.get("activate")
    fernet_key = os.getenv("FERNET_KEY")

    if activate_id:
        LLM.objects.update(is_active=False)
        LLM.objects.filter(id=activate_id).update(is_active=True)
        return redirect("llm_page")

    if request.method == "POST":
        name = request.POST.get("name")
        model = request.POST.get("model")
        api_key = request.POST.get("api_key")

        if name and model:
            fernet = Fernet(fernet_key)
            encrypted = fernet.encrypt(api_key.encode())
            LLM.objects.create(name=name, model=model, api_key=encrypted.decode())
        return redirect("llm_page")

    llms = LLM.objects.order_by("-created_at")

    return render(
        request,
        "llm_page.html",
        {
            "llm_name_choices": list(llm_constants.keys()),
            "llm_constants_json": json.dumps(llm_constants),
            "llms": llms,
        },
    )
