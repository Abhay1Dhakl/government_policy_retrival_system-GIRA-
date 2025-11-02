from django.db import models
from django.db.models import TextChoices


class DocumentType(TextChoices):
    # PI_LEAFLET = "PI_LEAFLET", "PIs, Leaflets"
    # LRD = "LRD", "LRDs"
    # GRD_ESCALATION = "GRD_ESCALATION", "GRDs+Escalations"
    # ONLINE_DATABASE = "ONLINE_DATABASE", "Online Databases-Embase, Pubmed"
    PI_LEAFLET = "pis"
    LRD = "lrd"
    GRD_ESCALATION = "grd"
    PAST_CASES = "past_cases"
    ONLINE_DB = "online_db", "Online Databases-Embase, Pubmed"

class SourceType(TextChoices):
    UPLOAD = "UPLOAD", "upload"
    API = "API", "api"


class FileSource(models.TextChoices):
    UPLOAD = "UPLOAD", "upload"
    ONEDRIVE = "ONEDRIVE", "onedrive"
    SHAREPOINT = "SHAREPOINT", "sharepoint"
    GOOGLE_DRIVE = "GOOGLE_DRIVE", "google_drive"
    SALESFORCE = "SALESFORCE", "salesforce"
    VEEVA = "VEEVA", "veeva"
