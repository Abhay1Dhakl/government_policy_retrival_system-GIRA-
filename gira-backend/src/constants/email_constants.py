from django.db import models


class EmailType:
    INVITE_NEW_USER = "invite_new_user"


class EmailStatus(models.TextChoices):
    PENDING = "Pending"
    SUCCESS = "Success"
    FAILED = "Failed"
