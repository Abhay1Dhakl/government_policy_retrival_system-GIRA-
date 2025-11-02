from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.conf.urls.static import static

from .views import DocumentViewSet


router = DefaultRouter(trailing_slash=True)
router.register(r"documents", DocumentViewSet, basename="documents")

urlpatterns = router.urls + static(
    settings.MEDIA_URL, document_root=settings.MEDIA_ROOT
)
