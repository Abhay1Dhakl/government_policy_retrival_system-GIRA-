from django.db import models
from src.constants.email_constants import EmailStatus


class MiraEmail(models.Model):
    class Meta:
        db_table = "mira_emails"
        app_label = "mira_emails"

    sender_email = models.EmailField()
    receiver_email = models.EmailField()
    email_type = models.CharField(max_length=50, null=True, blank=True)
    status = models.CharField(
        max_length=20, default=EmailStatus.PENDING, choices=EmailStatus.choices
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Email from {self.sender_email} to {self.receiver_email} - {self.email_type} - {self.status}"
