from rest_framework.routers import DefaultRouter

from .views.user import UserViewSet
from .views.password import PasswordViewSet
from .views.token import TokenViewSet


router = DefaultRouter(trailing_slash=True)
router.register(r"users", UserViewSet, basename="users")
router.register(r"password", PasswordViewSet, basename="password")
router.register(r"token", TokenViewSet, basename="token")

urlpatterns = router.urls
