from django.db import models


class LLM(models.Model):
    name = models.CharField(max_length=255)
    model = models.CharField(max_length=255)
    api_key = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
