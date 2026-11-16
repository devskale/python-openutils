from .base import CredgooBackend
from .gdrive import GdriveBackend
from .airtable import AirtableBackend
from .cfworker import CfworkerBackend

BACKENDS = {
    "gdrive": GdriveBackend,
    "airtable": AirtableBackend,
    "cfworker": CfworkerBackend,
}

BACKEND_LABELS = {
    "gdrive": "Google Drive (Apps Script)",
    "airtable": "Airtable",
    "cfworker": "Cloudflare Worker (self-hosted)",
}

__all__ = ["BACKENDS", "BACKEND_LABELS", "CredgooBackend", "GdriveBackend", "AirtableBackend", "CfworkerBackend"]
