from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'documents'
    
    def ready(self):
        try:
            import documents.signals

        except ImportError as e:
            print(f"Failed to load document signals: {e}")
   
