from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("users/", views.users_page, name="users_page"),
    path("users/delete/<int:user_id>/", views.user_delete, name="user_delete"),
    path("users/<int:user_id>/edit/", views.user_edit, name="user_edit"),
    path("documents/", views.documents_page, name="documents_page"),
    path("llm/", views.llm_page, name="llm_page"),
    path("documents/<int:doc_id>/", views.document_download, name="document_download"),
    path(
        "documents/delete/<int:doc_id>/", views.document_delete, name="document_delete"
    ),
    path(
        "documents/check-duplicate/",
        views.check_duplicate_document,
        name="check_duplicate_document",
    ),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
