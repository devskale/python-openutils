from .base import CredgooBackend
from .gdrive import GdriveBackend
from .airtable import AirtableBackend

BACKENDS = {
    "gdrive": GdriveBackend,
    "airtable": AirtableBackend,
}

BACKEND_LABELS = {
    "gdrive": "Google Drive (Apps Script)",
    "airtable": "Airtable",
}

__all__ = ["BACKENDS", "BACKEND_LABELS", "CredgooBackend", "GdriveBackend", "AirtableBackend"]
